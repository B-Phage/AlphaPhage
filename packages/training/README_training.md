# Notes on this the AlphaPhage trainer

AlphaPhage provides concise and MLOps-standard access to advanced AI methods for
predicting host-phage interactions, whether you have access to large cloud GPU
infrastructure, and requires sharding, or only have access to a laptop, making it
a flexible solution for phage therapy of the future in hospitals or research.
It was also developed with the disk space and speed constraints in mind by smart
data packaging technology and the h5 data format. Several innovations make the
AlphaPhage training framework extremely scalable. Data is pre-tokenised during
preprocessing, allowing for speed, and avoiding unneccessary compute of the same
strain during training. Host and phage are separately packaged in the h5 format,
and stored alongside pair keys and classes (infect/non-infect) that maps between
host and phage. During training and inference, host-phage paired samples are
reconstructed during each step in a memory-smart manner, and using the latest in
pytorch-lightning, following MLOps. This avoids an explosion of both the disk space
and memory required during training. Read access through this method using the
.hdf5 format is remarkably space efficient and fast. Futher, you get access to
the most widely used output formats and dashboards used in modern AI.


### Considerations before choosing a model


1. Alignment / Non-alignment. The feature_engineering package currently
implements alignment of features through bioinformatics workflows, and primarily
through genome annotation using profile-hidden markov models (PHMMs) - see
preprocessing. Alignment has a strong positive impact on predictive accuracy.

2. Encoding type. Data type currently is tokens. With models to be released, other
types of encoding will become available.

3. Data packing. The "reader_mode" sets the type of packing. using "binary_pairkey"
uses the AlphaPhage memory-smart dynamic sample building scheme.


### Models currently in AlphaPhage

1. TransformerX
contains several components, giving it the capabilities of a
Transformer Encoder, as well as those of a multilayer perceptron (MLP). Futher, it
harbours a unique, optional pre-compression step, in which case the architecture
can be referred to as a CompressedInputTransformer. Effectively,
the CompressedInputTransformer avoids the impact on memory requirement by
standard Transformers, and allows the compacting of a large feature space into
the available memory, whether you run on a cloud-based GPU system, or on a laptop.
TransformerX comes with a set of pre-configurations that allows a wide variety
of sub-architectures to be built. This extends the list of well-known
transformer-related hyperparameters to larger changes in the architecture.

### Configuration of a training run using TransformerX

Below is an example configuration file. Note in particular that archspec and
archspec2 are strings that specify sub-architectures. Both arcspec and arcspec2
consist of underscore-separated keys, specifying different parts of the TransformerX
architecture.

Parts of arcspec are described below:

Positional encoding:
  rope : rotary positional embedding
  nope : no positional embedding
Transformer:
  tt : torch_transformer
  nt : no transformer - implies MLP architecture.
Normalisation of embeddings post-transformer:
  nonormpostl : normalise
  nonormpostl : no normalization
Aggregation of embeddings post transformer:
  view : map each element of each embedding (dim1) to the MLP (dim2)
  cls : use the cls global token concept and map to MLP (dim2)
  sum : sum over embeddings (dim1) and map to MLP (dim2).

Parts of arcspec2 are described below:

Number of MLP layers (including first and last):
  L1 : one layer
  L2 : two layers
  L3 : three layers
  L4 : four layers

Methods of normalisation between layers in MLP:
bn : batch norm
ln : layernorm
nn : no normalization

At the bottom is a list of available settings. Any archspec string is compatible
with any archspec2 string. You could perform an exhaustive scan for optimised
sub-architectures along with other hyperparamers (see below).

arcspec strings:
  "rope_tt_normpostl_view"
  "rope_tt_normpostl_sum"
  "rope_tt_nonormpostl_view"
  "rope_tt_nonormpostl_sum"
  "rope_nt_normpostl_view"
  "rope_nt_normpostl_sum"
  "rope_nt_nonormpostl_view"
  "rope_nt_nonormpostl_sum"
  "nope_tt_normpostl_view"
  "nope_tt_normpostl_sum"
  "nope_tt_nonormpostl_view"
  "nope_tt_nonormpostl_sum"
  "nope_nt_normpostl_view"
  "nope_nt_normpostl_sum"
  "nope_nt_nonormpostl_view"
  "nope_nt_nonormpostl_sum"

arcspec2 strings:
  "L4_bn"
  "L4_ln"
  "L4_nn"
  "L3_bn"
  "L3_ln"
  "L3_nn"
  "L2_bn"
  "L2_ln"
  "L2_nn"
  "L1_nn"

Other hyperparameters:
  dim : Embedding dimension from nn.Embedding
  lcom : New context window length for CompressedInputTransformer mode
  dim1 : Embedding dimension used in transformer. Transformer configuration
    requires dim1=dim
  dim2 : Start dimension of downstream MLP layers (Aggregation may also
    involve linear layers)
  heads : Transformer heads
  mult : Transformer multiplier for inner dimension of FFN
  depth : Number of tranformer layers

A typical config file for a small test dataset would look like this, and located
at this location:

  config_path = "../../tests/unit/data"
  config_name = "config_test_2"
  and using the .yaml extension.

  seed  : 42
  verboselevel : "ERROR"
  debug_mode : false

  train_features_path : "../../bp/AlphaX/proc/sim_data/Tests/x_\<dataname\>.h5"
  train_labels_path : "../../bp/AlphaX/proc/sim_data/Tests/cl_\<dataname\>.h5"
  debug_features_path : ""
  debug_labels_path : ""

  pair_indices_name : "pair_indices"
  dataset_name_host : "host_vec_1"
  dataset_name_phage : "phage_vec_1"
  reader_mode : "binary_pairkey"
  architecture : "TransformerX"
  num_workers : 0

  trainer:
    logger_method : "MLFlowLogger"
    project : "AlphaPhage_Testing"
    run_name : "test_run_2"
    mlflow_tracking_uri : null
    epochs : 300
    accelerator : "cpu"
    devices : 1
    precision : 32
    batch_size : 10
    save_top_k : 3
    patience : 5
    val_fraction : 0.15

  hyp:
    archspec : "nope_nt_nonormpostl_sum"
    archspec2 : "L4_bn"
    dim : 8
    lcom : 30
    dim1 : 8
    dim2 : 8
    heads : 4
    mult : 4
    depth : 1
    vocab_size_out : 2

  optimizer:
    lr : 0.001

### Following a training run through loggers

Use the trainer.logger_method to choose how you would like to report training
results. Options are:

  "MLFlowLogger" : Allows tracking on MLFlow Browser through localhost or at a
  dedicated mlflow server, or results could be written to text files.
    For writing to text files, set trainer.mlflow_tracking_uri to
      http://<IP_ADDRESS>:5000.
    For casting to localhost, set trainer.mlflow_tracking_uri to
      http://localhost:5000/ and open your browser at http://localhost:5000/.
    For writing to text files, set trainer.mlflow_tracking_uri : null.

  "TensorBoardLogger" : Allows tracking on TensorboardX.
    Set trainer.mlflow_tracking_uri to "TensorBoardLogger".

  "WandbLogger" : Allows tracking on WandbLogger. You would sign up for an
    account on WandbLogger first.
    Set trainer.mlflow_tracking_uri to "WandbLogger".

  "CSVLogger" : Write a CSV file containing all metrics.
    Set trainer.mlflow_tracking_uri to "CSVLogger".
