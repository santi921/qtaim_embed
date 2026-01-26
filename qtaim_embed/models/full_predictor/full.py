"""
Full Predictor for iterative link-node prediction.

This module implements the iterative prediction loop that alternates between:
1. Link prediction (predicting edges/bonds)
2. Node prediction (predicting node features/properties)

The loop allows the graph topology and node features to be refined iteratively,
enabling prediction of QTAIM-style labeled graphs from molecular geometries.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import dgl
import numpy as np

from qtaim_embed.models.link_pred.link_model import GCNLinkPred
from qtaim_embed.models.node_level.base_gcn import GCNNodePred
from qtaim_embed.data.transforms import hetero_to_homo, homo_to_hetero
from qtaim_embed.data.geometry_to_graph import (
    GeometryToGraph,
    update_graph_topology,
    edges_from_predictions,
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
            - device: Device to run on ('cuda' or 'cpu')

    Example:
        >>> config = {
        ...     "link_model_path": "checkpoints/link_model.ckpt",
        ...     "node_model_path": "checkpoints/node_model.ckpt",
        ...     "iterations": 3,
        ...     "edge_threshold": 0.5,
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

        # Load pretrained models
        self.link_model = self._load_link_model(config["link_model_path"])
        self.node_model = self._load_node_model(config["node_model_path"])

        # Move models to device
        self.link_model = self.link_model.to(self.device)
        self.node_model = self.node_model.to(self.device)

        # Set models to eval mode
        self.link_model.eval()
        self.node_model.eval()

        # Transformer for hetero <-> homo conversion
        self.hetero_to_homo_transform = hetero_to_homo(concat_global=True)

    def _load_link_model(self, path: str) -> GCNLinkPred:
        """Load link prediction model from checkpoint."""
        try:
            model = GCNLinkPred.load_from_checkpoint(checkpoint_path=path)
            print(f"Link model loaded from {path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load link model from {path}: {e}")

    def _load_node_model(self, path: str) -> GCNNodePred:
        """Load node prediction model from checkpoint."""
        try:
            model = GCNNodePred.load_from_checkpoint(checkpoint_path=path)
            print(f"Node model loaded from {path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load node model from {path}: {e}")

    @torch.no_grad()
    def predict(
        self,
        graph: dgl.DGLHeteroGraph,
        return_intermediate: bool = False,
    ) -> Union[dgl.DGLHeteroGraph, Tuple[dgl.DGLHeteroGraph, List[Dict]]]:
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
        graph: dgl.DGLHeteroGraph,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Run link prediction on the graph.

        Converts heterograph to homograph, runs link prediction, and returns
        edge scores and predicted edge list.
        """
        # Convert to homogeneous graph for link prediction
        homo_graph = self.hetero_to_homo_transform(graph)
        homo_graph = homo_graph.to(self.device)

        # Get node features for link prediction
        node_feats = homo_graph.ndata["ft"]

        # For link prediction, we need to generate candidate edges
        # Use all possible edges as candidates
        num_nodes = homo_graph.num_nodes()
        candidate_edges = self._get_candidate_edges(num_nodes)

        # Create positive graph (current edges) and negative graph (non-edges)
        positive_graph = homo_graph

        # Create a candidate graph with all potential edges for scoring
        all_src = []
        all_dst = []
        for i, j in candidate_edges:
            all_src.append(i)
            all_dst.append(j)

        candidate_graph = dgl.graph(
            (all_src, all_dst),
            num_nodes=num_nodes,
        ).to(self.device)

        # Run link prediction
        # The model returns (pos_scores, neg_scores) - we use candidate_graph for both
        # to get scores for all candidate edges
        pos_scores, _ = self.link_model(positive_graph, candidate_graph, node_feats)

        # Convert scores to edge predictions using threshold
        edge_mask = torch.sigmoid(pos_scores) > self.edge_threshold
        predicted_edges = [
            candidate_edges[i] for i in range(len(candidate_edges))
            if edge_mask[i]
        ]

        return pos_scores, predicted_edges

    def _get_candidate_edges(self, num_nodes: int) -> List[Tuple[int, int]]:
        """Generate all candidate edges (upper triangular)."""
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edges.append((i, j))
        return edges

    def _update_topology(
        self,
        graph: dgl.DGLHeteroGraph,
        new_edges: List[Tuple[int, int]],
    ) -> dgl.DGLHeteroGraph:
        """
        Update graph topology with new predicted edges.

        Creates a new heterograph with the predicted edge structure while
        preserving atom features.
        """
        return update_graph_topology(graph, new_edges, self_loop=True)

    def _predict_nodes(
        self,
        graph: dgl.DGLHeteroGraph,
    ) -> Dict[str, torch.Tensor]:
        """
        Run node prediction on the graph.

        Returns predicted features for each node type in the target_dict.
        """
        # Get current node features
        node_feats = graph.ndata["feat"]

        # Run node prediction
        predictions = self.node_model(graph, node_feats)

        return predictions

    def _update_node_features(
        self,
        graph: dgl.DGLHeteroGraph,
        predictions: Dict[str, torch.Tensor],
    ) -> dgl.DGLHeteroGraph:
        """
        Update node features with predictions for next iteration.

        The predicted features are concatenated with or replace existing features
        depending on configuration.
        """
        # Update predicted node types with their predictions
        for node_type, pred in predictions.items():
            if node_type in graph.ntypes:
                # Get current features
                current_feat = graph.nodes[node_type].data.get("feat")

                if current_feat is not None:
                    # Option 1: Replace features with predictions
                    # This assumes the prediction size matches what we need
                    if pred.shape[1] <= current_feat.shape[1]:
                        # Update only the predicted dimensions
                        new_feat = current_feat.clone()
                        new_feat[:, :pred.shape[1]] = pred
                        graph.nodes[node_type].data["feat"] = new_feat
                    else:
                        # Predictions are larger - extend features
                        graph.nodes[node_type].data["feat"] = pred

                # Store predictions separately for final output
                graph.nodes[node_type].data["pred"] = pred

        return graph

    def predict_from_geometry(
        self,
        coords: np.ndarray,
        elements: List[str],
        charge: int = 0,
        distance_cutoff: float = 1.8,
        **kwargs,
    ) -> dgl.DGLHeteroGraph:
        """
        Convenience method to predict directly from geometry.

        Args:
            coords: Atomic coordinates of shape (N, 3).
            elements: List of element symbols.
            charge: Molecular charge.
            distance_cutoff: Initial distance cutoff for edge creation.
            **kwargs: Additional arguments passed to predict().

        Returns:
            Labeled heterograph with predicted structure and properties.
        """
        # Create initial graph from geometry
        converter = GeometryToGraph(distance_cutoff=distance_cutoff)
        initial_graph = converter(coords, elements, charge=charge)

        # Run prediction
        return self.predict(initial_graph, **kwargs)

    def evaluate(
        self,
        graph: dgl.DGLHeteroGraph,
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
                if node_type in predicted_graph.ntypes:
                    pred_feat = predicted_graph.nodes[node_type].data.get("pred")
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
            },
            "node_model": {
                "type": type(self.node_model).__name__,
                "n_conv_layers": self.node_model.hparams.get("n_conv_layers"),
                "hidden_size": self.node_model.hparams.get("hidden_size"),
                "target_dict": self.node_model.hparams.get("target_dict"),
            },
            "config": {
                "iterations": self.iterations,
                "edge_threshold": self.edge_threshold,
            },
        }


class FullPredictorInference:
    """
    Simplified inference-only wrapper for FullPredictor.

    This class provides a cleaner interface for production use when you just
    need to run predictions without training capabilities.

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
        geometry_converter: GeometryToGraph,
    ):
        self.full_predictor = full_predictor
        self.geometry_converter = geometry_converter

    @classmethod
    def from_checkpoints(
        cls,
        link_ckpt: str,
        node_ckpt: str,
        iterations: int = 3,
        edge_threshold: float = 0.5,
        distance_cutoff: float = 1.8,
        device: str = "cpu",
    ) -> "FullPredictorInference":
        """
        Create inference predictor from checkpoint paths.

        Args:
            link_ckpt: Path to link model checkpoint.
            node_ckpt: Path to node model checkpoint.
            iterations: Number of prediction iterations.
            edge_threshold: Threshold for edge predictions.
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
            "device": device,
        }
        full_predictor = FullPredictor(config)
        geometry_converter = GeometryToGraph(distance_cutoff=distance_cutoff)

        return cls(full_predictor, geometry_converter)

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
        for node_type in predicted_graph.ntypes:
            if "pred" in predicted_graph.nodes[node_type].data:
                result[f"{node_type}_predictions"] = (
                    predicted_graph.nodes[node_type].data["pred"].cpu().numpy()
                )

        return result


# Backward compatibility alias
fullPredictor = FullPredictor
