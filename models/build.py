import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg):
    
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)

    ## TODO: Not implemented yet
    if cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        load_checkpoint(model, cfg.TRAIN.CHECKPOINT_FILE_PATH, strict = False)
    
    return model