""" """

# import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import utils.consol_log as log
from arc.transformer_classifier import TransformerXClassifier
from omegaconf import DictConfig, OmegaConf

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
    ):
        super().__init__()
        self.cfg = cfg

        # Save hyperparameters as dict
        hparams = OmegaConf.to_container(cfg, resolve=True)
        self.save_hyperparameters(hparams)

        self.archspecs = archspecs
        self.archspecs2 = archspecs2

        # Define pre-configuration based on archspecs
        self.positional_encoding = self.archspecs[self.cfg.hyp.archspec]["positional_encoding"]
        self.layer_method = self.archspecs[self.cfg.hyp.archspec]["layer_method"]
        self.norm_post_layers = self.archspecs[self.cfg.hyp.archspec]["norm_post_layers"]
        self.aggr_attention_method = self.archspecs[self.cfg.hyp.archspec]["aggr_attention_method"]

        # Define pre-configuration based on archspecs2
        self.call = self.archspecs2[self.cfg.hyp.archspec2]["call"]
        self.n = self.archspecs2[self.cfg.hyp.archspec2]["n"]
        self.dim = self.cfg.hyp.dim
        self.lcom = self.cfg.hyp.lcom
        self.dim1 = self.cfg.hyp.dim1
        self.dim2 = self.cfg.hyp.dim2
        self.heads = self.cfg.hyp.heads
        self.mult = self.cfg.hyp.mult
        self.depth = self.cfg.hyp.depth

        # Perform architecture setting checks
        if self.positional_encoding not in ["rope", None]:
            # logger.error(f"Error: self.positional_encoding {self.positional_encoding} not recognised. Exiting.")
            print(f"Error: self.positional_encoding '{self.positional_encoding}' not recognised.")
            raise ValueError(
                f"Error: self.positional_encoding '{self.positional_encoding}' not recognised."
            )
        if self.layer_method not in ["torch_transformer", None]:
            # logger.error(f"Error: self.layer_method {self.layer_method} not recognised. Exiting.")
            print(f"Error: self.layer_method '{self.layer_method}' not recognised.")
            raise ValueError(f"Error: self.layer_method '{self.layer_method}' not recognised.")
        if self.norm_post_layers not in [True, False]:
            # logger.error(f"Error: self.norm_post_layers {self.norm_post_layers} not recognised. Exiting.")
            print(f"Error: self.norm_post_layers '{self.norm_post_layers}' not recognised.")
            raise ValueError(
                f"Error: self.norm_post_layers '{self.norm_post_layers}' not recognised."
            )
        if self.aggr_attention_method not in ["view", "sum", "cls_token"]:
            # logger.error(f"Error: self.aggr_attention_method {self.aggr_attention_method} not recognised. Exiting.")
            print(
                f"Error: self.aggr_attention_method '{self.aggr_attention_method}' not recognised."
            )
            raise ValueError(
                f"Error: self.aggr_attention_method '{self.aggr_attention_method}' not recognised."
            )
        if self.call not in ["L4", "L3", "L2", "L1"]:
            # logger.error(f"Error: self.call {self.call} not recognised. Exiting.")
            print(f"Error: self.call '{self.call}' not recognised.")
            raise ValueError(f"Error: self.call '{self.call}' not recognised.")
        if self.n not in ["bn", "ln", "nn"]:
            # logger.error(f"Error: self.n {self.n} not recognised. Exiting.")
            print(f"Error: self.n '{self.n}' not recognised.")
            raise ValueError(f"Error: self.n '{self.n}' not recognised.")

        if self.depth == 0 and self.layer_method is not None:
            print("Error: self.depth == 0 and self.layer_method is not None")
            raise ValueError(f"Error: self.depth == 0 and self.layer_method is not None")

        if self.layer_method is not None and self.dim1 != self.dim:
            print(f"self.layer_method is not None and self.dim1 != self.dim.")
            raise ValueError(f"self.layer_method is not None and self.dim1 != self.dim.")

        self.model = TransformerXClassifier(
            positional_encoding=self.positional_encoding,
            layer_method=self.layer_method,
            norm_post_layers=self.norm_post_layers,
            aggr_attention_method=self.aggr_attention_method,
            call=self.call,
            n=self.n,
            dim=self.cfg.hyp.dim,
            lcom=self.cfg.hyp.lcom,
            dim1=self.cfg.hyp.dim1,
            dim2=self.cfg.hyp.dim2,
            heads=self.cfg.hyp.heads,
            mult=self.cfg.hyp.mult,
            depth=self.cfg.hyp.depth,
            num_tokens=num_tokens,
            vocab_size_out=vocab_size_out,
            length_padded=length_padded,
        )

    def forward(self, x):
        return self.model(x)  # .logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        train_loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        val_loss = torch.nn.functional.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        return optimizer
