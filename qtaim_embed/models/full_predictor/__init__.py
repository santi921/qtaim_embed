"""
Full Predictor module for iterative link-node prediction.

This module provides the FullPredictor class that combines pretrained link prediction
and node prediction models to iteratively predict molecular graph structure and properties.
"""

from qtaim_embed.models.full_predictor.full import FullPredictor

__all__ = ["FullPredictor"]
