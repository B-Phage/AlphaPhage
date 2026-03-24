"""TransformerX module for training a Transformer model using PyTorch Lightning."""

from typing import cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import utils.consol_log as log
from models.transformer_classifier import TransformerXClassifier
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import BinaryRecall, BinarySpecificity

logger = log.get_logger(level=log.WARNING)


class TransformerX(pl.LightningModule):
    """
    A PyTorch Lightning Module for training a Transformer model.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_tokens: int,
        vocab_size_out: int,
        length_padded: int,
        archspecs: dict,
        archspecs2: dict,
        class_weights=None,
    ):
        super().__init__()

        # 1. Resolve the config so paths and variables are absolute strings/ints
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

        # 2. Save the resolved config AND the standalone metadata
        # We pass them as a dict to ensure everything is captured in one go
        self.save_hyperparameters(
            {
                "cfg": resolved_cfg,
                "num_tokens": num_tokens,
                "vocab_size_out": vocab_size_out,
                "length_padded": length_padded,
            }
        )

        self.cfg = cfg
        self.archspecs = archspecs
        self.archspecs2 = archspecs2

        # Define pre-configuration based on archspecs
        self.positional_encoding = self.archspecs[self.cfg.archspec]["positional_encoding"]
        self.layer_method = self.archspecs[self.cfg.archspec]["layer_method"]
        self.norm_post_layers = self.archspecs[self.cfg.archspec]["norm_post_layers"]
        self.aggr_attention_method = self.archspecs[self.cfg.archspec]["aggr_attention_method"]

        # Define pre-configuration based on archspecs2
        self.call = self.archspecs2[self.cfg.archspec2]["call"]
        self.n = self.archspecs2[self.cfg.archspec2]["n"]
        self.dim = self.cfg.dim
        self.lcom = self.cfg.lcom
        self.dim1 = self.cfg.dim1
        self.dim2 = self.cfg.dim2
        self.heads = self.cfg.heads
        self.mult = self.cfg.mult
        self.depth = self.cfg.depth

        # Perform architecture setting checks
        if self.positional_encoding not in ["rope", None]:
            raise ValueError(
                f"Error: self.positional_encoding '{self.positional_encoding}' not recognised."
            )
        if self.layer_method not in ["torch_transformer", None]:
            raise ValueError(f"Error: self.layer_method '{self.layer_method}' not recognised.")
        if self.norm_post_layers not in [True, False]:
            raise ValueError(
                f"Error: self.norm_post_layers '{self.norm_post_layers}' not recognised."
            )
        if self.aggr_attention_method not in ["view", "sum", "cls_token"]:
            raise ValueError(
                f"Error: self.aggr_attention_method '{self.aggr_attention_method}' not recognised."
            )
        if self.n not in ["bn", "ln", "nn"]:
            raise ValueError(f"Error: self.n '{self.n}' not recognised.")

        if self.depth == 0 and self.layer_method is not None:
            raise ValueError(f"Error: self.depth == 0 and self.layer_method is not None")

        if self.layer_method is not None and self.dim1 != self.dim:
            raise ValueError(f"self.layer_method is not None and self.dim1 != self.dim.")

        self.model = TransformerXClassifier(
            positional_encoding=self.positional_encoding,
            layer_method=self.layer_method,
            norm_post_layers=self.norm_post_layers,
            aggr_attention_method=self.aggr_attention_method,
            call=self.call,
            n=self.n,
            dim=self.cfg.dim,
            lcom=self.cfg.lcom,
            dim1=self.cfg.dim1,
            dim2=self.cfg.dim2,
            heads=self.cfg.heads,
            mult=self.cfg.mult,
            depth=self.cfg.depth,
            num_tokens=num_tokens,
            vocab_size_out=vocab_size_out,
            length_padded=length_padded,
        )

        # Register the weights so they move to GPU/CPU automatically
        if class_weights is not None:
            logger.info(f"TransformerX assigned class weights of: {class_weights}")
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", None)

        # Initialize metrics
        # We use Binary versions assuming your 80/20 data is a binary classification
        self.val_sensitivity = BinaryRecall()
        self.val_specificity = BinarySpecificity()

    def forward(self, x):
        return self.model(x)  # .logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        # Cast self.class_weights to torch.Tensor to satisfy Pylance
        weights = cast(torch.Tensor, self.class_weights) if self.class_weights is not None else None

        train_loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        # Cast self.class_weights to torch.Tensor to satisfy Pylance
        weights = cast(torch.Tensor, self.class_weights) if self.class_weights is not None else None

        val_loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)

        # Update and Log Metrics
        # Setting prog_bar=True is what makes them appear in your terminal
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        sens = self.val_sensitivity(preds, y)
        spec = self.val_specificity(preds, y)

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val/sens", sens, on_epoch=True, prog_bar=True)
        self.log("val/spec", spec, on_epoch=True, prog_bar=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        test_loss = torch.nn.functional.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        return test_loss

    def predict_step(self, batch, batch_idx):
        # function might only be provided with x (features) without y (labels).
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(x)
        return torch.softmax(logits, dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Space saving override
        # Remove the optimizer states and other heavy metadata
        keys_to_remove = ["optimizer_states", "lr_schedulers", "loops", "callbacks"]
        for key in keys_to_remove:
            checkpoint.pop(key, None)
