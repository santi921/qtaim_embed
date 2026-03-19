"""
Full Predictor for iterative link-node prediction.

This module implements the iterative prediction loop that alternates between:
1. Link prediction (predicting edges/bonds)
2. Node prediction (predicting node features/properties)

The loop allows the graph topology and node features to be refined iteratively,
enabling prediction of QTAIM-style labeled graphs from molecular geometries.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
import torch

logger = logging.getLogger(__name__)
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, HeteroData

from qtaim_embed.models.link_pred.link_model import GCNLinkPred
from qtaim_embed.models.node_level.base_gcn import GCNNodePred
from qtaim_embed.data.transforms import hetero_to_homo
from qtaim_embed.data.geometry_to_graph import (
    GeometryToGraph,
    update_graph_topology,
    edges_from_predictions,
)
from qtaim_embed.models.utils import (
    load_link_model_from_config,
    load_node_level_model_from_config,
    get_grapher_config_from_model,
)


class FullPredictor(nn.Module):
    """
    Full predictor that combines link and node prediction in an iterative loop.

    This class loads pretrained link and node prediction models and runs them
    iteratively to predict both the graph topology (edges) and node properties
    from an initial molecular geometry.

    Args:
        config: Configuration dictionary with the following keys:
            - link_model_path: Path to pretrained link model checkpoint
            - node_model_path: Path to pretrained node model checkpoint
            - iterations: Number of prediction iterations (default: 3)
            - edge_threshold: Threshold for link prediction scores (default: 0.5)
            - edge_aggregation: How to aggregate bidirectional edge scores, either
              "max" or "avg" (default: "max"). This handles asymmetric predictors
              like MLPPredictor by scoring both (i,j) and (j,i) then aggregating.
            - device: Device to run on ('cuda' or 'cpu')

    Example:
        >>> config = {
        ...     "link_model_path": "checkpoints/link_model.ckpt",
        ...     "node_model_path": "checkpoints/node_model.ckpt",
        ...     "iterations": 3,
        ...     "edge_threshold": 0.5,
        ...     "edge_aggregation": "max",
        ... }
        >>> predictor = FullPredictor(config)
        >>> labeled_graph = predictor.predict(initial_graph)
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.iterations = config.get("iterations", 3)
        self.edge_threshold = config.get("edge_threshold", 0.5)
        self.device = config.get("device", "cpu")
        # How to aggregate bidirectional edge scores: "max" or "avg"
        self.edge_aggregation = config.get("edge_aggregation", "max")
        if self.edge_aggregation not in ("max", "avg"):
            raise ValueError(f"edge_aggregation must be 'max' or 'avg', got {self.edge_aggregation}")

        # Load pretrained models
        self.link_model = self._load_link_model(config["link_model_path"])
        logger.info("Link model loaded.")
        logger.debug("Link input model: %s", self.link_model.hparams.input_size)
        self.node_model = self._load_node_model(config["node_model_path"])
        logger.info("Node model loaded.")
        logger.debug("Node input model (atom): %s", self.node_model.hparams.atom_input_size)
        logger.debug("Node input model (bond): %s", self.node_model.hparams.bond_input_size)
        logger.debug("Node input model (global): %s", self.node_model.hparams.global_input_size)

        # Move models to device
        self.link_model = self.link_model.to(self.device)
        self.node_model = self.node_model.to(self.device)

        # Set models to eval mode
        self.link_model.eval()
        self.node_model.eval()

        # Extract grapher config from models for consistent featurization
        self.grapher_config = self._get_grapher_config_from_models()

        # Create geometry converter with matching element_set
        distance_cutoff = config.get("distance_cutoff", 1.8)
        if self.grapher_config is not None:
            logger.info(f"Using element_set from model: {self.grapher_config.get('element_set', [])[:5]}...")
            self.geometry_converter = GeometryToGraph.from_grapher_config(
                self.grapher_config,
                distance_cutoff=distance_cutoff,
            )
        else:
            warnings.warn(
                "Models lack grapher_config - using default element set. "
                "For consistent featurization, retrain models with grapher_config."
            )
            self.geometry_converter = GeometryToGraph(
                distance_cutoff=distance_cutoff,
            )

        # Transformer for hetero <-> homo conversion
        self.hetero_to_homo_transform = hetero_to_homo(concat_global=True)

    def _load_link_model(self, path: str) -> GCNLinkPred:
        """Load link prediction model from checkpoint."""
        try:
            model = GCNLinkPred.load_from_checkpoint(checkpoint_path=path)
            logger.info(f"Link model loaded from {path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load link model from {path}: {e}")

    def _load_node_model(self, path: str) -> GCNNodePred:
        """Load node prediction model from checkpoint."""
        try:
            model = GCNNodePred.load_from_checkpoint(checkpoint_path=path)
            logger.info(f"Node model loaded from {path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load node model from {path}: {e}")

    def _get_grapher_config_from_models(self) -> Optional[Dict]:
        """
        Extract grapher configuration from loaded models.

        Tries the node model first (has full hetero graph featurizer info),
        then falls back to the link model.

        Returns:
            dict or None: The grapher configuration, or None if not present in either model.
        """
        # Try node model first (has full hetero featurizer info)
        config = get_grapher_config_from_model(self.node_model)
        if config is not None:
            return config

        # Fall back to link model (may have limited info)
        return get_grapher_config_from_model(self.link_model)

    @torch.no_grad()
    def predict(
        self,
        graph: HeteroData,
        return_intermediate: bool = False,
    ) -> Union[HeteroData, Tuple[HeteroData, List[Dict]]]:
        """
        Run the iterative prediction loop on a graph.

        Args:
            graph: Initial heterograph with atom features. Can be created using
                GeometryToGraph or from existing data.
            return_intermediate: If True, return intermediate results from each
                iteration along with the final graph.

        Returns:
            Final heterograph with predicted edges and node features.
            If return_intermediate=True, also returns list of intermediate results.
        """
        graph = graph.to(self.device)
        intermediate_results = []

        for iteration in range(self.iterations):
            # Step 1: Link prediction
            edge_scores, predicted_edges = self._predict_links(graph)

            # Step 2: Update graph topology based on link predictions
            graph = self._update_topology(graph, predicted_edges)

            # Step 3: Node prediction
            node_predictions = self._predict_nodes(graph)

            # Step 4: Update node features (for next iteration)
            graph = self._update_node_features(graph, node_predictions)

            if return_intermediate:
                intermediate_results.append({
                    "iteration": iteration,
                    "edge_scores": edge_scores,
                    "predicted_edges": predicted_edges,
                    "node_predictions": {k: v.clone() for k, v in node_predictions.items()},
                    "graph": graph.clone() if hasattr(graph, 'clone') else None,
                })

        if return_intermediate:
            return graph, intermediate_results
        return graph

    def _predict_links(
        self,
        graph: HeteroData,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Run link prediction on the graph.

        Converts heterograph to homograph, runs link prediction, and returns
        edge scores and predicted edge list.

        Note: Scores both directions (i->j and j->i) for each edge pair to handle
        asymmetric predictors (e.g., MLPPredictor). Aggregates using max or avg
        based on self.edge_aggregation config.
        """
        # Convert to homogeneous graph for link prediction
        homo_graph = self.hetero_to_homo_transform(graph)
        homo_graph = homo_graph.to(self.device)

        # Get node features for link prediction
        node_feats = homo_graph.ft

        # For link prediction, we need to generate candidate edges
        # Generate bidirectional edges to handle asymmetric predictors
        num_nodes = homo_graph.num_nodes
        bidirectional_edges = self._get_bidirectional_candidate_edges(num_nodes)

        # Create positive graph (current edges)
        positive_graph = homo_graph

        # Create a candidate graph with all potential edges for scoring
        all_src = [e[0] for e in bidirectional_edges]
        all_dst = [e[1] for e in bidirectional_edges]

        candidate_graph = Data(
            edge_index=torch.tensor([all_src, all_dst], dtype=torch.long),
            num_nodes=num_nodes,
        ).to(self.device)

        # Run link prediction
        # The model returns (pos_scores, neg_scores) - we use candidate_graph for both
        # to get scores for all candidate edges
        pos_scores, _ = self.link_model(positive_graph, candidate_graph, node_feats)

        # Aggregate bidirectional scores to get one score per undirected edge
        aggregated_scores, canonical_edges = self._aggregate_bidirectional_scores(
            pos_scores, bidirectional_edges, num_nodes
        )

        # Convert scores to edge predictions using threshold
        edge_probs = torch.sigmoid(aggregated_scores)
        edge_mask = edge_probs > self.edge_threshold
        predicted_edges = [
            canonical_edges[i] for i in range(len(canonical_edges))
            if edge_mask[i]
        ]

        return aggregated_scores, predicted_edges

    def _get_bidirectional_candidate_edges(self, num_nodes: int) -> List[Tuple[int, int]]:
        """
        Generate all candidate edges in both directions.

        For each pair (i, j) where i != j, generates both (i, j) and (j, i).
        This ensures compatibility with asymmetric predictors like MLPPredictor.
        """
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append((i, j))
        return edges

    def _aggregate_bidirectional_scores(
        self,
        scores: torch.Tensor,
        bidirectional_edges: List[Tuple[int, int]],
        num_nodes: int,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Aggregate scores from both directions into a single score per undirected edge.

        Args:
            scores: Tensor of shape (num_bidirectional_edges,) with scores for each directed edge
            bidirectional_edges: List of (src, dst) tuples corresponding to scores
            num_nodes: Number of nodes in the graph

        Returns:
            aggregated_scores: Tensor of shape (num_undirected_edges,)
            canonical_edges: List of (i, j) tuples where i < j
        """
        # Build a mapping from directed edge to score
        edge_to_score = {}
        for idx, (src, dst) in enumerate(bidirectional_edges):
            edge_to_score[(src, dst)] = scores[idx]

        # Generate canonical edges (i < j) and aggregate scores
        canonical_edges = []
        aggregated_scores = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                canonical_edges.append((i, j))
                score_ij = edge_to_score.get((i, j), torch.tensor(float('-inf')))
                score_ji = edge_to_score.get((j, i), torch.tensor(float('-inf')))

                if self.edge_aggregation == "max":
                    agg_score = torch.max(score_ij, score_ji)
                else:  # avg
                    agg_score = (score_ij + score_ji) / 2.0

                aggregated_scores.append(agg_score)

        return torch.stack(aggregated_scores), canonical_edges

    def _update_topology(
        self,
        graph: HeteroData,
        new_edges: List[Tuple[int, int]],
    ) -> HeteroData:
        """
        Update graph topology with new predicted edges.

        Creates a new heterograph with the predicted edge structure while
        preserving atom features.
        """
        return update_graph_topology(graph, new_edges, self_loop=True)

    def _predict_nodes(
        self,
        graph: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """
        Run node prediction on the graph.

        Returns predicted features for each node type in the target_dict.
        """
        # Get current node features as dict expected by node model
        node_feats = {ntype: graph[ntype].feat for ntype in graph.node_types}

        # Run node prediction
        predictions = self.node_model(graph, node_feats)

        return predictions

    def _update_node_features(
        self,
        graph: HeteroData,
        predictions: Dict[str, torch.Tensor],
    ) -> HeteroData:
        """
        Update node features with predictions for next iteration.

        The predicted features are concatenated with or replace existing features
        depending on configuration.
        """
        # Update predicted node types with their predictions
        for node_type, pred in predictions.items():
            if node_type in graph.node_types:
                # Get current features
                current_feat = getattr(graph[node_type], 'feat', None)

                if current_feat is not None:
                    # Option 1: Replace features with predictions
                    # This assumes the prediction size matches what we need
                    if pred.shape[1] <= current_feat.shape[1]:
                        # Update only the predicted dimensions
                        new_feat = current_feat.clone()
                        new_feat[:, :pred.shape[1]] = pred
                        graph[node_type].feat = new_feat
                    else:
                        # Predictions are larger - extend features
                        graph[node_type].feat = pred

                # Store predictions separately for final output
                graph[node_type].pred = pred

        return graph

    def predict_from_geometry(
        self,
        coords: np.ndarray,
        elements: List[str],
        charge: int = 0,
        **kwargs,
    ) -> HeteroData:
        """
        Convenience method to predict directly from geometry.

        Uses the geometry converter configured at initialization, which has the
        correct element_set from the model's grapher_config for consistent
        featurization.

        Args:
            coords: Atomic coordinates of shape (N, 3).
            elements: List of element symbols.
            charge: Molecular charge.
            **kwargs: Additional arguments passed to predict().

        Returns:
            Labeled heterograph with predicted structure and properties.
        """
        # Create initial graph using the configured geometry converter
        initial_graph = self.geometry_converter(coords, elements, charge=charge)

        # Run prediction
        return self.predict(initial_graph, **kwargs)

    def evaluate(
        self,
        graph: HeteroData,
        ground_truth_edges: Optional[List[Tuple[int, int]]] = None,
        ground_truth_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate prediction quality against ground truth.

        Args:
            graph: Input graph to predict on.
            ground_truth_edges: True edge list for edge prediction evaluation.
            ground_truth_features: True node features for node prediction evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}

        # Run prediction
        predicted_graph, intermediate = self.predict(graph, return_intermediate=True)

        # Evaluate edge prediction if ground truth provided
        if ground_truth_edges is not None:
            final_edges = intermediate[-1]["predicted_edges"]
            gt_set = set(tuple(sorted(e)) for e in ground_truth_edges)
            pred_set = set(tuple(sorted(e)) for e in final_edges)

            # Compute precision, recall, F1
            true_positives = len(gt_set & pred_set)
            precision = true_positives / len(pred_set) if pred_set else 0.0
            recall = true_positives / len(gt_set) if gt_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics["edge_precision"] = precision
            metrics["edge_recall"] = recall
            metrics["edge_f1"] = f1

        # Evaluate node features if ground truth provided
        if ground_truth_features is not None:
            for node_type, gt_feat in ground_truth_features.items():
                if node_type in predicted_graph.node_types:
                    pred_feat = getattr(predicted_graph[node_type], 'pred', None)
                    if pred_feat is not None:
                        # Ensure same shape
                        min_dim = min(pred_feat.shape[1], gt_feat.shape[1])
                        pred_feat = pred_feat[:, :min_dim]
                        gt_feat = gt_feat[:, :min_dim]

                        # Compute MAE
                        mae = torch.mean(torch.abs(pred_feat - gt_feat)).item()
                        metrics[f"{node_type}_mae"] = mae

                        # Compute MSE
                        mse = torch.mean((pred_feat - gt_feat) ** 2).item()
                        metrics[f"{node_type}_mse"] = mse

        return metrics

    def get_model_info(self) -> Dict:
        """Return information about the loaded models."""
        return {
            "link_model": {
                "type": type(self.link_model).__name__,
                "n_conv_layers": self.link_model.hparams.get("n_conv_layers"),
                "hidden_size": self.link_model.hparams.get("hidden_size"),
                "predictor": self.link_model.hparams.get("predictor"),
                "grapher_config": self.link_model.hparams.get("grapher_config"),
            },
            "node_model": {
                "type": type(self.node_model).__name__,
                "n_conv_layers": self.node_model.hparams.get("n_conv_layers"),
                "hidden_size": self.node_model.hparams.get("hidden_size"),
                "target_dict": self.node_model.hparams.get("target_dict"),
                "grapher_config": self.node_model.hparams.get("grapher_config"),
            },
            "config": {
                "iterations": self.iterations,
                "edge_threshold": self.edge_threshold,
                "edge_aggregation": self.edge_aggregation,
            },
            "grapher_config": self.grapher_config,
        }


class FullPredictorInference:
    """
    Simplified inference-only wrapper for FullPredictor.

    This class provides a cleaner interface for production use when you just
    need to run predictions without training capabilities. The geometry converter
    is automatically configured from the models' grapher_config for consistent
    element encoding.

    Example:
        >>> predictor = FullPredictorInference.from_checkpoints(
        ...     link_ckpt="link_model.ckpt",
        ...     node_ckpt="node_model.ckpt",
        ... )
        >>> result = predictor(coords, elements)
    """

    def __init__(
        self,
        full_predictor: FullPredictor,
    ):
        self.full_predictor = full_predictor
        # Use the geometry converter from FullPredictor, which has the correct element_set
        self.geometry_converter = full_predictor.geometry_converter

    @classmethod
    def from_checkpoints(
        cls,
        link_ckpt: str,
        node_ckpt: str,
        iterations: int = 3,
        edge_threshold: float = 0.5,
        edge_aggregation: str = "max",
        distance_cutoff: float = 1.8,
        device: str = "cpu",
    ) -> "FullPredictorInference":
        """
        Create inference predictor from checkpoint paths.

        The geometry converter is automatically configured from the models'
        grapher_config to ensure consistent element one-hot encoding.

        Args:
            link_ckpt: Path to link model checkpoint.
            node_ckpt: Path to node model checkpoint.
            iterations: Number of prediction iterations.
            edge_threshold: Threshold for edge predictions.
            edge_aggregation: How to aggregate bidirectional scores ("max" or "avg").
            distance_cutoff: Distance cutoff for initial graph.
            device: Device to run on.

        Returns:
            FullPredictorInference instance.
        """
        config = {
            "link_model_path": link_ckpt,
            "node_model_path": node_ckpt,
            "iterations": iterations,
            "edge_threshold": edge_threshold,
            "edge_aggregation": edge_aggregation,
            "distance_cutoff": distance_cutoff,  # Pass to FullPredictor
            "device": device,
        }
        full_predictor = FullPredictor(config)

        return cls(full_predictor)

    def __call__(
        self,
        coords: np.ndarray,
        elements: List[str],
        charge: int = 0,
        spin_multiplicity: int = 1,
    ) -> Dict:
        """
        Run prediction from coordinates and elements.

        Args:
            coords: Atomic coordinates of shape (N, 3).
            elements: List of element symbols.
            charge: Molecular charge.
            spin_multiplicity: Spin multiplicity.

        Returns:
            Dictionary with predicted edges and node features.
        """
        # Create initial graph
        initial_graph = self.geometry_converter(
            coords, elements, charge=charge, spin_multiplicity=spin_multiplicity
        )

        # Run prediction
        predicted_graph, intermediate = self.full_predictor.predict(
            initial_graph, return_intermediate=True
        )

        # Extract results
        final_edges = intermediate[-1]["predicted_edges"]

        result = {
            "predicted_edges": final_edges,
            "num_predicted_bonds": len(final_edges),
            "graph": predicted_graph,
        }

        # Extract predicted node features
        for node_type in predicted_graph.node_types:
            pred = getattr(predicted_graph[node_type], 'pred', None)
            if pred is not None:
                result[f"{node_type}_predictions"] = pred.cpu().numpy()

        return result


# Backward compatibility alias
fullPredictor = FullPredictor
