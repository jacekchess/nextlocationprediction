from fvcore.common.config import CfgNode

_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #

_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True
_C.TRAIN.DATASET = 'Sensible'
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.MAX_EPOCHS = 20
# Evaluate model on test data every eval period epochs.
#_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
#_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
#_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# If True, reset epochs when loading checkpoint.
# _C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# ---------------------------------------------------------------------------- #
# Test options.
# ---------------------------------------------------------------------------- #

_C.TEST = CfgNode()
_C.TEST.ENABLE = True
_C.TEST.DATASET = 'Sensible'
_C.TEST.BATCH_SIZE = 32
_C.TEST.SAVE_RESULTS_PATH = ""

# ---------------------------------------------------------------------------- #
# Model options.
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.MODEL_NAME = "LSTM"
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.LOSS_FUNC = "CrossEntropyLoss"
_C.MODEL.DROPOUT_RATE = 0.5


# ---------------------------------------------------------------------------- #
# Data options.
# ---------------------------------------------------------------------------- #
_C.DATA = CfgNode()
_C.DATA.PATH_TO_DATA_DIR = ""

# ---------------------------------------------------------------------------- #
# Solver options.
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.LR = 0.1
_C.SOLVER.OPTIMIZING_METHOD = "adam"
_C.SOLVER.WEIGHT_DECAY = 0.00001
_C.SOLVER.LR_MILESTONES = [3,5]
_C.SOLVER.LR_GAMMA = 0.1

# ---------------------------------------------------------------------------- #
# Misc options.
# ---------------------------------------------------------------------------- #
_C.NUM_GPUS = 1
_C.ACCU_BATCH = 1
_C.OUTPUT_DIR = ""

# ---------------------------------------------------------------------------- #
# Data loader options.
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# LSTM options.
# ---------------------------------------------------------------------------- #
_C.LSTM = CfgNode()
_C.LSTM.embedding_dim = 13
_C.LSTM.hidden_dim = 15
_C.LSTM.n_loc_rank = 1000
_C.LSTM.n_loc_type = 17040
_C.LSTM.n_layers = 3
_C.LSTM.cut_off = 20
_C.LSTM.stop_t = 926
_C.LSTM.seq_len = 100
_C.LSTM.train_percentage = 0.85

#----------------------------------------------------------------------------#
# Deprecated.
#----------------------------------------------------------------------------#
_C.BN = CfgNode()
_C.BN.WEIGHT_DECAY = 0.0001

def _assert_and_infer_cfg(cfg):
    
    # TRAIN assertions
    cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions
    cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    return cfg

def get_cfg():
    return _assert_and_infer_cfg(_C.clone())