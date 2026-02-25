# dataloaders

This package provides utilities and classes for loading, preprocessing, and managing datasets in AlphaPhage projects.

## Features
The package dataloaders implements MLOps-standard data loading, directly from hdf5
format. The BinaryPairkeyFeatureDataset object builds each sample on-the-fly as
a host and phage pair from pre-tokenised data. This method enables a massive feature
set to be used as the host-phage pairs do not have to be built up-front, which would
consume a large amount of disk space (number of pairs = number of host * number of
phages).

## Installation
```
make setup-dataloaders
```
