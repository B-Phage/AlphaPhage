# kaptive3api

This package wraps the kaptive3 tool for characterising the kapsid of bacteria.
The kaptive code base is found here https://github.com/klebgenomics/Kaptive/tree/master.
The kaptive toobase was originally built for Klebsiella, but can be adapted for
other gram negative bacteria. Some groups have published other species-specific
database files that can be used by kapsid as replacement for the default Klebsiella
database, for example, that for Escherichia coli -
https://github.com/rgladstone/EC-K-typing/tree/main/DB
We have found that the kaptive outputs are an important part of the feature set
of the host for host-phage match prediction on E.coli and its phages.

## Features
The kaptive tool uses Blast and currently requiring the python 2.9 and run in the
Intel emulation environment.

## Installation
```
make setup-kaptive3api
```
Note that in the Make file, the installation of the environment using
make setup-kaptive3api induces the Intel emulation (--subdir osx-64),
```
conda env create --subdir osx-64 -f conda_envs/$1.yml -n $1;
```
and fixing the environment as the emulation environment by subsequently running
```
conda config --env --set subdir osx-64.
```
The latter step is required to ensure that any future ```conda install``` commands
recognises the Intel environment.
