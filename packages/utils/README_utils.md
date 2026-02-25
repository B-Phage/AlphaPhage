# utils

This package provides utilities for getting vocabularies and configs.

## Features
get_vocab: obtain a vocabulary as OmegaConf object.
get_config: get params as OmegaConf object. Note that as sub-directory is retrieved
based on params_set.
get_config_generic: get params as OmegaConf object, with extended methods of use.

## Examples:
Get whole config straigth from train_v0.0.1.yaml
```
cfg = get_config_generic(path=args.config)
```
Get trainer sub-config straigth from train_v0.0.1.yaml
```
cfg = get_config_generic(path=args.config, params_set="trainer")
```
Get trainer sub-config straigth from train_v0.0.1.yaml
```
cfg = get_config_generic(path=args.config, params_set=["trainer"])
```
get path from params.yaml
```
cfg = get_config_generic(path='params.yaml', params_set=["train","train_config"])
```
get path from params.yaml and then load config from that path and the "trainer" sub-config
```
cfg = get_config_generic(path='params.yaml',
                            params_set=["train","train_config"],
                            params_set_2=["trainer"])
```
Get path from params.yaml and then load config from that path and the "trainer" sub-config and withing it, the batch_size
```
cfg = get_config_generic(path='params.yaml',
                            params_set=["train","train_config"],
                            params_set_2=["trainer","batch_size"])
```

## Installation
```
make setup-utils
```
