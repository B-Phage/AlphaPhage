"""
'Transformer' :
    input: integer tokens.
    embedding > transformer blocks > linear classifier
"""

import torch
import torch.nn.functional as F
import utils.consol_log as log
from torch import Tensor, nn
from training.embeddings import RotaryPositionalEmbeddings

logger = log.get_logger(level=log.WARNING)


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
        positional_encoding: str,
        layer_method: str,
        norm_post_layers: bool,
        aggr_attention_method: str,
        call: str,
        n: str,
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
    ):

        super().__init__()
        self.positional_encoding = positional_encoding
        self.layer_method = layer_method
        self.norm_post_layers = norm_post_layers
        self.aggr_attention_method = aggr_attention_method
        self.call = call
        self.n = n
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

        logger.info(f"building Embedding layer...")
        self.embed = nn.Embedding(self.num_tokens, self.dim, padding_idx=0)
        logger.info(f"building attention layers...")
        if self.positional_encoding == "rope":
            self.rope_encoding = RotaryPositionalEmbeddings(
                dim=self.dim, max_seq_len=self.length_padded
            )

        if self.lcom is not None:
            self.LinCom = nn.Linear(self.length_padded * self.dim, self.lcom * self.dim1)

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

        # Parse number of layers from "L4", "L3", "L2", "L1", etc.
        self.num_mlp_layers = int(self.call[1:])

        # Build MLP Layers Dynamically
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_mlp_layers):
            is_last = i == self.num_mlp_layers - 1

            # Determine output dimension
            out_features = self.vocab_size_out if is_last else self.dim2
            self.linears.append(nn.Linear(self.dim2, out_features))

            # Add Norm layers only for hidden (non-last) layers
            if not is_last:
                if self.n == "bn":
                    self.norms.append(nn.BatchNorm1d(self.dim2))
                elif self.n == "ln":
                    self.norms.append(nn.LayerNorm(self.dim2))
                else:
                    self.norms.append(nn.Identity())

    def forward(self, x):
        """Default train."""

        x = self.embed(x)

        if self.positional_encoding == "rope":
            x = self.rope_encoding(x).squeeze(0)  # Rotary embeddig
        elif self.positional_encoding == None:
            pass

        # Compression of length (CompresedInputTranformer)
        if self.lcom != None:
            x = x.view(-1, self.length_padded * self.dim)  # flatten last 2 dims
            x = self.LinCom(x)  # length_padded*dim -> lcom*dim1
            x = x.view(-1, self.lcom, self.dim1)  # lcom*dim1 -> lcom,dim1

        if self.aggr_attention_method == "cls_token":
            cls_token_expanded = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token_expanded, x], dim=1)

        if self.layer_method == None:
            pass

        elif self.layer_method == "torch_transformer":
            x = self.transformer_encoder(x)

        if self.norm_post_layers == True:
            x = self.ln(x)
        elif self.norm_post_layers == False:
            pass

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

        # MLP Head
        for i in range(self.num_mlp_layers - 1):
            # Linear layers
            x = self.linears[i](x)

            # Apply Norm (Index matches because we only created N-1 norms)
            x = self.norms[i](x)

            # Standard ReLU activation for hidden layers
            x = F.relu(x)

        # Final Output Layer (No Norm, No ReLU)
        # This gives you the raw logits for your Loss function
        x = self.linears[-1](x)

        return x
