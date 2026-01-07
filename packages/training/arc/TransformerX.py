"""
'Transformer' :
    input: integer tokens.
    embedding > transformer blocks > linear classifier
"""

import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import utils.consol_log as log
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from training.embeddings import RotaryPositionalEmbeddings

logger = log.get_logger(level=log.WARNING)

archspecs = {
    "rope_tt_normpostl_view": {
        "positional_encoding": "rope",
        "layer_method": "torch_transformer",
        "norm_post_layers": False,
        "aggr_attention_method": "view",
    },
    "rope_tt_normpostl_sum": {
        "positional_encoding": "rope",
        "layer_method": "torch_transformer",
        "norm_post_layers": False,
        "aggr_attention_method": "sum",
    },
    "rope_tt_normpostl_cls": {
        "positional_encoding": "rope",
        "layer_method": "torch_transformer",
        "norm_post_layers": False,
        "aggr_attention_method": "cls_token",
    },
    "rope_tt_nonormpostl_view": {
        "positional_encoding": "rope",
        "layer_method": "torch_transformer",
        "norm_post_layers": True,
        "aggr_attention_method": "view",
    },
    "rope_tt_nonormpostl_sum": {
        "positional_encoding": "rope",
        "layer_method": "torch_transformer",
        "norm_post_layers": True,
        "aggr_attention_method": "sum",
    },
    "rope_tt_nonormpostl_cls": {
        "positional_encoding": "rope",
        "layer_method": "torch_transformer",
        "norm_post_layers": True,
        "aggr_attention_method": "cls_token",
    },
    "rope_nt_normpostl_view": {
        "positional_encoding": "rope",
        "layer_method": None,
        "norm_post_layers": False,
        "aggr_attention_method": "view",
    },
    "rope_nt_normpostl_sum": {
        "positional_encoding": "rope",
        "layer_method": None,
        "norm_post_layers": False,
        "aggr_attention_method": "sum",
    },
    "rope_nt_normpostl_cls": {
        "positional_encoding": "rope",
        "layer_method": None,
        "norm_post_layers": False,
        "aggr_attention_method": "cls_token",
    },
    "rope_nt_nonormpostl_view": {
        "positional_encoding": "rope",
        "layer_method": None,
        "norm_post_layers": True,
        "aggr_attention_method": "view",
    },
    "rope_nt_nonormpostl_sum": {
        "positional_encoding": "rope",
        "layer_method": None,
        "norm_post_layers": True,
        "aggr_attention_method": "sum",
    },
    "rope_nt_nonormpostl_cls": {
        "positional_encoding": "rope",
        "layer_method": None,
        "norm_post_layers": True,
        "aggr_attention_method": "cls_token",
    },
    "nope_tt_normpostl_view": {
        "positional_encoding": None,
        "layer_method": "torch_transformer",
        "norm_post_layers": False,
        "aggr_attention_method": "view",
    },
    "nope_tt_normpostl_sum": {
        "positional_encoding": None,
        "layer_method": "torch_transformer",
        "norm_post_layers": False,
        "aggr_attention_method": "sum",
    },
    "nope_tt_normpostl_cls": {
        "positional_encoding": None,
        "layer_method": "torch_transformer",
        "norm_post_layers": False,
        "aggr_attention_method": "cls_token",
    },
    "nope_tt_nonormpostl_view": {
        "positional_encoding": None,
        "layer_method": "torch_transformer",
        "norm_post_layers": True,
        "aggr_attention_method": "view",
    },
    "nope_tt_nonormpostl_sum": {
        "positional_encoding": None,
        "layer_method": "torch_transformer",
        "norm_post_layers": True,
        "aggr_attention_method": "sum",
    },
    "nope_tt_nonormpostl_cls": {
        "positional_encoding": None,
        "layer_method": "torch_transformer",
        "norm_post_layers": True,
        "aggr_attention_method": "cls_token",
    },
    "nope_nt_normpostl_view": {
        "positional_encoding": None,
        "layer_method": None,
        "norm_post_layers": False,
        "aggr_attention_method": "view",
    },
    "nope_nt_normpostl_sum": {
        "positional_encoding": None,
        "layer_method": None,
        "norm_post_layers": False,
        "aggr_attention_method": "sum",
    },
    "nope_nt_normpostl_cls": {
        "positional_encoding": None,
        "layer_method": None,
        "norm_post_layers": False,
        "aggr_attention_method": "cls_token",
    },
    "nope_nt_nonormpostl_view": {
        "positional_encoding": None,
        "layer_method": None,
        "norm_post_layers": True,
        "aggr_attention_method": "view",
    },
    "nope_nt_nonormpostl_sum": {
        "positional_encoding": None,
        "layer_method": None,
        "norm_post_layers": True,
        "aggr_attention_method": "sum",
    },
    "nope_nt_nonormpostl_cls": {
        "positional_encoding": None,
        "layer_method": None,
        "norm_post_layers": True,
        "aggr_attention_method": "cls_token",
    },
}
archspecs2 = {
    "L4_bn": {"call": "L4", "n": "bn"},
    "L4_ln": {"call": "L4", "n": "ln"},
    "L4_nn": {"call": "L4", "n": "nn"},
    "L3_bn": {"call": "L3", "n": "bn"},
    "L3_ln": {"call": "L3", "n": "ln"},
    "L3_nn": {"call": "L3", "n": "nn"},
    "L2_bn": {"call": "L2", "n": "bn"},
    "L2_ln": {"call": "L2", "n": "ln"},
    "L2_nn": {"call": "L2", "n": "nn"},
    "L1_nn": {"call": "L1", "n": "nn"},
}
archspec_list = [
    "rope_tt_normpostl_view",
    "rope_tt_normpostl_sum",
    "rope_tt_nonormpostl_view",
    "rope_tt_nonormpostl_sum",
    "rope_nt_normpostl_view",
    "rope_nt_normpostl_sum",
    "rope_nt_nonormpostl_view",
    "rope_nt_nonormpostl_sum",
    "nope_tt_normpostl_view",
    "nope_tt_normpostl_sum",
    "nope_tt_nonormpostl_view",
    "nope_tt_nonormpostl_sum",
    "nope_nt_normpostl_view",
    "nope_nt_normpostl_sum",
    "nope_nt_nonormpostl_view",
    "nope_nt_nonormpostl_sum",
]
archspec2_list = [
    "L4_bn",
    "L4_ln",
    "L4_nn",
    "L3_bn",
    "L3_ln",
    "L3_nn",
    "L2_bn",
    "L2_ln",
    "L2_nn",
    "L1_nn",
]


class TransformerXClassifier(nn.Module):
    """Builds a Transformer-based classifier.

    Keyword arguments:
    archspecs -- Aa dict defined in this module with archspec as keys
    archspec -- Will determine: "positional_encoding", "layer_method", \
        "norm_post_layers", "aggr_attention_method".
    archspecs2 -- A dict defined in this module with archspec2 as keys
    archspec2 -- Will determine: "call", "n"
    dim -- Embedding dimension from nn.Embedding
    lcom -- New context window length for CompressedInputTransformer mode
    dim1 -- Embedding dimension used in transformer. Transformer configuration \
        requires dim1=dim
    dim2 -- Start dimension of downstream MLP layers (Aggregation may also \
        involve linear layers)
    heads -- Transformer heads
    mult -- Transformer multiplier for inner dimension of FFN
    depth -- Number of tranformer layers
    num_tokens  -- When running nn.Embedding
    vocab_size_out -- Output dimension of x for this nn
    length_padded -- Input length (number of tokens per sample, padded)


    Return: A Transformer-based classifier model.
    """

    def __init__(
        self,
        archspecs: dict,
        archspec: str,
        archspecs2: dict,
        archspec2: str,
        dim: int,
        lcom: int,
        dim1: int,
        dim2: int,
        heads: int,
        mult: int,
        depth: int,
        num_tokens: int,
        vocab_size_out: int,
        length_padded: int,
        # *args,
        # **kwargs,
    ):

        super().__init__()
        self.archspecs = archspecs
        self.archspec = archspec
        self.archspecs2 = archspecs2
        self.archspec2 = archspec2
        self.dim = dim
        self.lcom = lcom
        self.dim1 = dim1
        self.dim2 = dim2
        self.heads = heads
        self.mult = mult
        self.depth = depth
        self.num_tokens = num_tokens
        self.vocab_size_out = vocab_size_out
        self.length_padded = length_padded

        # Define pre-configuration based on archspecs
        self.positional_encoding = self.archspecs[self.archspec]["positional_encoding"]
        self.layer_method = self.archspecs[self.archspec]["layer_method"]
        self.norm_post_layers = self.archspecs[self.archspec]["norm_post_layers"]
        self.aggr_attention_method = self.archspecs[self.archspec]["aggr_attention_method"]
        # Define pre-configuration based on archspecs2
        self.call = self.archspecs2[self.archspec2]["call"]
        self.n = self.archspecs2[self.archspec2]["n"]

        logger.info(f"building Embedding layer...")
        self.embed = nn.Embedding(self.num_tokens, self.dim, padding_idx=0)
        logger.info(f"building attention layers...")

        if self.positional_encoding == "rope":
            # from embeddings import RotaryPositionalEmbeddings

            self.rope_encoding = RotaryPositionalEmbeddings(
                dim=self.dim, max_seq_len=self.length_padded
            )
        elif self.positional_encoding == None:
            pass

        if self.layer_method != None and self.dim1 != self.dim:
            # transformer mode
            logger.error(
                f"Error: dim1 {self.dim1} != dim {self.dim} \
                         and lcom == None (no input copression applied on \
                         context window)."
            )
            sys.exit()

        if self.lcom == None:
            # not CompressedInputTransformer mode
            pass
        else:  # CompressedInputTransformer mode.
            self.LinCom = nn.Linear(self.length_padded * self.dim, self.lcom * self.dim1)
            # length_padded*dim -> lcom*self.dim1

        if self.depth == 0 and self.layer_method != None:
            # Not wanting Transformer layers.
            logger.error(
                f"Error: depth was set to 0 and layer_method == \
                         {self.layer_method}. Rather set layer_method = None \
                            if using 0. Exiting."
            )
            sys.exit()

        if self.aggr_attention_method == "cls_token":
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim1))

        # dimension is dim1
        if self.layer_method == "torch_transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.dim1,
                nhead=self.heads,
                dim_feedforward=self.dim1 * self.mult,
                dropout=0,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)

        elif self.layer_method == None:
            self.dim1 = self.dim  # Apply aggregation from same as dim1

        if self.aggr_attention_method == "view":
            if self.norm_post_layers == True:
                self.ln = nn.LayerNorm(self.dim1)
            if self.lcom == None:
                # Transformer mode
                self.Linaggr = nn.Linear(self.length_padded * self.dim1, self.dim2)
            else:
                # CompressedInputTransformer mode
                self.Linaggr = nn.Linear(self.lcom * self.dim1, self.dim2)

        elif self.aggr_attention_method in ["sum", "cls_token"]:
            if self.norm_post_layers == True:
                self.ln = nn.LayerNorm(self.dim1)
            self.Linaggr = nn.Linear(self.dim1, self.dim2)

        elif self.aggr_attention_method == "cls_token":
            pass

        else:
            logger.error(
                f"Error: self.aggr_attention_method \
                         {self.aggr_attention_method} not recognised. Exiting."
            )
            sys.exit()

        if self.call not in ["L4", "L3", "L2", "L1"]:
            logger.error(f"Error: self.call {self.call} not recognised. Exiting.")
            sys.exit()
        if self.n not in ["bn", "ln", "nn"]:
            logger.error(f"Error: self.n {self.n} not recognised. Exiting.")
            sys.exit()

        if self.call == "L4":
            self.Lin1 = nn.Linear(self.dim2, self.dim2)
            self.Lin2 = nn.Linear(self.dim2, self.dim2)
            self.Lin3 = nn.Linear(self.dim2, self.dim2)
            self.Lin4 = nn.Linear(self.dim2, self.vocab_size_out)
            if self.n == "bn":
                self.bn1 = nn.BatchNorm1d(self.dim2)
                self.bn2 = nn.BatchNorm1d(self.dim2)
                self.bn3 = nn.BatchNorm1d(self.dim2)
            elif self.n == "ln":
                self.ln1 = nn.LayerNorm(self.dim2)
                self.ln2 = nn.LayerNorm(self.dim2)
                self.ln3 = nn.LayerNorm(self.dim2)

        elif self.call == "L3":
            self.Lin1 = nn.Linear(self.dim2, self.dim2)
            self.Lin2 = nn.Linear(self.dim2, self.dim2)
            self.Lin3 = nn.Linear(self.dim2, self.vocab_size_out)
            if self.n == "bn":
                self.bn1 = nn.BatchNorm1d(self.dim2)
                self.bn2 = nn.BatchNorm1d(self.dim2)
            elif self.n == "ln":
                self.ln1 = nn.LayerNorm(self.dim2)
                self.ln2 = nn.LayerNorm(self.dim2)

        elif self.call == "L2":
            self.Lin1 = nn.Linear(self.dim2, self.dim2)
            self.Lin2 = nn.Linear(self.dim2, self.vocab_size_out)
            if self.n == "bn":
                self.bn1 = nn.BatchNorm1d(self.dim2)
            elif self.n == "ln":
                self.ln1 = nn.LayerNorm(self.dim2)

        elif self.call == "L1":
            self.Lin1 = nn.Linear(self.dim2, self.vocab_size_out)

        else:
            logger.error(f"Error: self.call {self.call} not recognised. Exiting.")
            sys.exit()

    def forward(self, x):
        """output_mode : (train|score). Deafult train."""
        logger.info(f"x input: {x.size()}")

        x = self.embed(x)
        logger.info(f"x after embed: {x.size()}")

        if self.positional_encoding == "rope":
            x = self.rope_encoding(x).squeeze(0)  # Rotary embeddig
        elif self.positional_encoding == None:
            pass
        else:
            logger.error(
                f"Error: positional_encoding {self.positional_encoding} \
                         not recognised. Exiting."
            )
            sys.exit()

        # Compression of length (CompresedInputTranformer)
        if self.lcom != None:
            x = x.view(-1, self.length_padded * self.dim)  # flatten last 2 dims
            x = self.LinCom(x)  # length_padded*dim -> lcom*dim1
            x = x.view(-1, self.lcom, self.dim1)  # lcom*dim1 -> lcom,dim1

        if self.aggr_attention_method == "cls_token":
            # The [CLS] token will aggregate the sequence information for
            # classification
            cls_token_expanded = self.cls_token.expand(x.shape[0], -1, -1)
            # add the token as the first embedding in all samples
            x = torch.cat([cls_token_expanded, x], dim=1)

        if self.layer_method == None:
            pass

        elif self.layer_method == "torch_transformer":
            x = self.transformer_encoder(x)
            logger.info(f"x after layers: {x.size()}")

        else:
            logger.error(
                f"Error: layer_method {self.layer_method} not \
                         recognised. Exiting."
            )
            sys.exit()

        if self.norm_post_layers == True:
            x = self.ln(x)
        elif self.norm_post_layers == False:
            pass
        else:
            logger.error(
                f"Error: norm_post_layers {self.norm_post_layers} not \
                         recognised. Exiting."
            )
            sys.exit()

        d1, d2, d3 = x.shape

        if self.aggr_attention_method == "view":
            x = x.view(d1, d2 * d3)
            x = self.Linaggr(x)  # dim1->dim2
            x = F.silu(x)

        elif self.aggr_attention_method == "sum":
            x = x.sum(dim=-2)
            x = self.Linaggr(x)  # dim1->dim2 ##
            x = F.silu(x)

        elif self.aggr_attention_method == "cls_token":
            x = x[:, 0, :]  # Discard rest of x - just use CLS token.
            x = self.Linaggr(x)  # dim1->dim2
            x = F.silu(x)

        else:
            logger.error(
                f"Error: aggr_attention_method \
                         {self.aggr_attention_method} not recognised. Exiting."
            )
            sys.exit()

        if self.call == "L4":
            identity = x
            x = self.Lin1(x)
            if self.n == "bn":
                x = self.bn1(x)
            elif self.n == "ln":
                x = self.ln1(x)
            x = F.silu(x) + identity
            logger.info(f"x after Lin1: {x.size()}")
            identity = x
            x = self.Lin2(x)
            if self.n == "bn":
                x = self.bn2(x)
            elif self.n == "ln":
                x = self.ln2(x)
            x = F.silu(x) + identity
            logger.info(f"x after Lin2: {x.size()}")
            identity = x
            x = self.Lin3(x)
            if self.n == "bn":
                x = self.bn3(x)
            elif self.n == "ln":
                x = self.ln3(x)
            x = F.silu(x) + identity
            logger.info(f"x after Lin3: {x.size()}")
            x = self.Lin4(x)
            logger.info(f"x after Lin4: {x.size()}")

        elif self.call == "L3":
            identity = x
            x = self.Lin1(x)
            if self.n == "bn":
                x = self.bn1(x)
            elif self.n == "ln":
                x = self.ln1(x)
            x = F.silu(x) + identity
            logger.info(f"x after Lin1: {x.size()}")
            identity = x
            x = self.Lin2(x)
            if self.n == "bn":
                x = self.bn2(x)
            elif self.n == "ln":
                x = self.ln2(x)
            x = F.silu(x) + identity
            logger.info(f"x after Lin2: {x.size()}")
            x = self.Lin3(x)
            logger.info(f"x after Lin3: {x.size()}")

        elif self.call == "L2":
            identity = x
            x = self.Lin1(x)
            if self.n == "bn":
                x = self.bn1(x)
            elif self.n == "ln":
                x = self.ln1(x)
            x = F.silu(x) + identity
            logger.info(f"x after Lin1: {x.size()}")
            x = self.Lin2(x)
            logger.info(f"x after Lin2: {x.size()}")

        elif self.call == "L1":
            x = self.Lin1(x)
            x = F.silu(x)
            logger.info(f"x after Lin1: {x.size()}")

        else:
            logger.error(f"Error: call {self.call} not recognised. Exiting.")
            sys.exit()

        logger.info(f"x after linear layers: {x.size()}")

        return x


class TransformerX(pl.LightningModule):
    """
    A PyTorch Lightning Module for training a Transformer model.
    """

    def __init__(self, cfg: DictConfig, num_tokens: int, vocab_size_out: int, length_padded: int):
        super().__init__()
        self.cfg = cfg

        # Save hyperparameters as dict
        if isinstance(cfg, DictConfig):
            hparams = OmegaConf.to_container(cfg, resolve=True)
        else:
            hparams = cfg
        self.save_hyperparameters(hparams)

        # from arc.TransformerX import TransformerXClassifier
        # from embeddings import RotaryPositionalEmbeddings

        self.model = TransformerXClassifier(
            archspecs=archspecs,
            archspec=self.cfg.hyp.archspec,
            archspecs2=archspecs2,
            archspec2=self.cfg.hyp.archspec2,
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        return optimizer
