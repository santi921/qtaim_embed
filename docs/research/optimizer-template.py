"""
Optimizer Configuration Template for PyTorch Lightning Models

This template shows best practices for implementing configure_optimizers()
with fused optimizer support, graceful fallbacks, and proper scheduler setup.

Copy this pattern to new models to avoid optimizer overhead issues.

Author: qtaim_embed development team
Date: 2026-02-05
Reference: docs/research/optimizer-best-practices.md
"""

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from typing import Dict, List, Union, Optional, Any


class OptimizedLightningModel(pl.LightningModule):
    """
    Template model demonstrating optimizer best practices.

    This template includes:
    - Fused optimizer with graceful fallback
    - Parameter filtering
    - Flexible optimizer selection
    - Learning rate scheduler support
    - Proper logging and monitoring
    """

    def __init__(
        self,
        # Model architecture params
        input_size: int = 128,
        hidden_size: int = 256,
        output_size: int = 1,
        # Optimizer params
        optimizer: str = "adam",  # "adam", "adamw", "sgd"
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_fused: bool = True,  # Enable fused optimizer
        # Scheduler params
        scheduler_name: str = "reduce_on_plateau",  # "reduce_on_plateau", "none"
        lr_plateau_patience: int = 5,
        lr_scale_factor: float = 0.5,
        # Other
        **kwargs,
    ):
        super().__init__()

        # Save all hyperparameters
        self.save_hyperparameters()

        # Example model architecture (replace with your own)
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """Forward pass (replace with your logic)."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step (replace with your logic)."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step (replace with your logic)."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    # =========================================================================
    # OPTIMIZER CONFIGURATION - TEMPLATE SECTION
    # Copy this section to your models
    # =========================================================================

    def configure_optimizers(self):
        """
        Configure optimizer with fused support and scheduler.

        This implementation includes:
        1. Parameter filtering (only trainable params)
        2. Fused optimizer with graceful fallback
        3. Flexible optimizer selection
        4. Learning rate scheduler support
        5. Proper logging for debugging

        Returns:
            Union[torch.optim.Optimizer, Dict]: Optimizer or dict with scheduler
        """

        # Step 1: Filter parameters (only optimize trainable params)
        params = filter(lambda p: p.requires_grad, self.parameters())

        # Step 2: Select optimizer class
        optimizer_class = self._get_optimizer_class()

        # Step 3: Build optimizer arguments
        optimizer_kwargs = {
            "lr": self.hparams.lr,
            "weight_decay": self.hparams.weight_decay,
        }

        # Step 4: Try to use fused optimizer if requested and supported
        optimizer = self._create_optimizer(
            optimizer_class=optimizer_class,
            params=params,
            optimizer_kwargs=optimizer_kwargs,
        )

        # Step 5: Add scheduler if requested
        if self.hparams.scheduler_name.lower() != "none":
            scheduler = self._create_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # Metric to monitor
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer

    def _get_optimizer_class(self) -> type:
        """
        Get optimizer class from config string.

        Returns:
            type: PyTorch optimizer class
        """
        optimizer_map = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }

        optimizer_name = self.hparams.optimizer.lower()
        if optimizer_name not in optimizer_map:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. "
                f"Choose from: {list(optimizer_map.keys())}"
            )

        return optimizer_map[optimizer_name]

    def _create_optimizer(
        self,
        optimizer_class: type,
        params,
        optimizer_kwargs: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        """
        Create optimizer with fused support and fallback.

        Args:
            optimizer_class: PyTorch optimizer class
            params: Model parameters to optimize
            optimizer_kwargs: Optimizer configuration

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        # Check if fused is supported and requested
        fused_supported = self._optimizer_supports_fused(optimizer_class)

        if self.hparams.use_fused and fused_supported:
            # Try fused first
            try:
                optimizer = optimizer_class(
                    params,
                    **optimizer_kwargs,
                    fused=True,  # GPU-accelerated
                )
                # Optional: Log success (useful for debugging)
                if hasattr(self, 'log_text'):
                    self.log_text("optimizer/type", "fused")
                # print(f"✅ Using fused {optimizer_class.__name__}")
                return optimizer

            except RuntimeError as e:
                # Fallback to regular optimizer
                # print(f"⚠️  Fused optimizer unavailable: {e}")
                # print(f"   Falling back to regular {optimizer_class.__name__}")
                pass  # Fall through to regular optimizer

        # Create regular optimizer (fallback or by choice)
        optimizer = optimizer_class(params, **optimizer_kwargs)

        # Optional: Log optimizer type
        if hasattr(self, 'log_text'):
            self.log_text("optimizer/type", "regular")

        return optimizer

    def _optimizer_supports_fused(self, optimizer_class: type) -> bool:
        """
        Check if optimizer supports fused implementation.

        Args:
            optimizer_class: PyTorch optimizer class

        Returns:
            bool: True if fused is supported
        """
        # As of PyTorch 2.0+, Adam and AdamW support fused
        fused_optimizers = [torch.optim.Adam, torch.optim.AdamW]
        return optimizer_class in fused_optimizers

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer

        Returns:
            Optional scheduler instance
        """
        scheduler_name = self.hparams.scheduler_name.lower()

        if scheduler_name == "reduce_on_plateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.lr_scale_factor,
                patience=self.hparams.lr_plateau_patience,
                verbose=True,
            )

        elif scheduler_name == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                verbose=True,
            )

        elif scheduler_name == "step":
            return lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=self.hparams.lr_scale_factor,
                verbose=True,
            )

        elif scheduler_name == "none":
            return None

        else:
            raise ValueError(
                f"Unsupported scheduler: {scheduler_name}. "
                f"Choose from: reduce_on_plateau, cosine, step, none"
            )


# =============================================================================
# MINIMAL TEMPLATE (for quick copy-paste)
# =============================================================================

class MinimalOptimizedModel(pl.LightningModule):
    """
    Minimal template with just the essentials.

    Copy this for simple models where you just need fused optimizer.
    """

    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(10, 1)  # Replace with your model

    def configure_optimizers(self):
        """Configure optimizer with fused support."""
        # Filter parameters
        params = filter(lambda p: p.requires_grad, self.parameters())

        # Try fused optimizer with fallback
        try:
            optimizer = torch.optim.Adam(
                params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                fused=True,  # 20-40% faster
            )
        except RuntimeError:
            # Fallback to regular Adam
            optimizer = torch.optim.Adam(
                params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

        return optimizer


# =============================================================================
# ADVANCED TEMPLATE (with monitoring and parameter groups)
# =============================================================================

class AdvancedOptimizedModel(pl.LightningModule):
    """
    Advanced template with parameter groups and monitoring.

    Use this when you need:
    - Different learning rates for different layers
    - Detailed optimizer monitoring
    - Weight decay only for specific layers
    """

    def __init__(
        self,
        lr: float = 1e-3,
        lr_backbone: float = 1e-4,  # Lower LR for pretrained backbone
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Example: Pretrained backbone + new head
        self.backbone = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, 1)

    def configure_optimizers(self):
        """
        Configure optimizer with parameter groups.

        This allows different learning rates and settings for different
        parts of the model (e.g., lower LR for pretrained backbone).
        """

        # Define parameter groups with different settings
        param_groups = [
            {
                "params": self.backbone.parameters(),
                "lr": self.hparams.lr_backbone,  # Lower LR
                "weight_decay": 0.0,  # No weight decay for backbone
            },
            {
                "params": self.head.parameters(),
                "lr": self.hparams.lr,  # Normal LR
                "weight_decay": self.hparams.weight_decay,
            },
        ]

        # Try fused optimizer
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                fused=True,
            )
        except RuntimeError:
            optimizer = torch.optim.AdamW(param_groups)

        # Add scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_epoch_start(self):
        """Log learning rates at start of each epoch."""
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(f"lr/group_{i}", param_group["lr"])


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_usage():
    """Example usage of the templates."""

    # Example 1: Simple model with fused optimizer
    model = MinimalOptimizedModel(lr=0.001)

    # Example 2: Model with scheduler
    model = OptimizedLightningModel(
        input_size=128,
        hidden_size=256,
        output_size=1,
        optimizer="adam",
        lr=0.001,
        weight_decay=0.0,
        use_fused=True,  # Enable fused
        scheduler_name="reduce_on_plateau",
    )

    # Example 3: Model with parameter groups
    model = AdvancedOptimizedModel(
        lr=0.001,
        lr_backbone=0.0001,  # Lower LR for backbone
        weight_decay=0.01,
    )

    # Train with Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        profiler="pytorch",  # Enable profiling
    )

    # trainer.fit(model, train_dataloader, val_dataloader)


# =============================================================================
# TESTING TEMPLATE
# =============================================================================

def test_optimizer_template():
    """
    Test template to verify optimizer configuration.

    Add this to your test suite to catch optimizer issues early.
    """
    import pytest

    # Test 1: Verify fused optimizer on CUDA
    if torch.cuda.is_available():
        model = MinimalOptimizedModel().cuda()
        opt = model.configure_optimizers()

        # Extract optimizer from potential dict/list
        if isinstance(opt, dict):
            opt = opt["optimizer"]
        if isinstance(opt, list):
            opt = opt[0]

        # Check fused parameter
        assert "fused" in opt.defaults, "Fused parameter not found"
        assert opt.defaults["fused"] == True, "Fused not enabled"
        print("✅ Fused optimizer test passed")

    # Test 2: Verify CPU fallback
    model = MinimalOptimizedModel().cpu()
    opt = model.configure_optimizers()  # Should not crash
    print("✅ CPU fallback test passed")

    # Test 3: Verify parameter filtering
    model = OptimizedLightningModel()
    # Freeze some parameters
    for param in model.model[0].parameters():
        param.requires_grad = False

    opt = model.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]

    # Count optimized parameters
    optimized_params = sum(1 for group in opt.param_groups for p in group["params"])
    total_params = sum(1 for p in model.parameters())

    assert optimized_params < total_params, "Frozen parameters not filtered"
    print("✅ Parameter filtering test passed")


if __name__ == "__main__":
    print("="*60)
    print("Optimizer Template Test")
    print("="*60)

    print("\nRunning tests...")
    test_optimizer_template()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

    print("\nUsage:")
    print("1. Copy the template class to your model file")
    print("2. Replace the model architecture with your own")
    print("3. Keep the configure_optimizers() method as-is")
    print("4. Run profiling to verify speedup")
    print("\nSee docs/research/optimizer-best-practices.md for details")
