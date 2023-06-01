# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 8
# Path to dataset, could be overwritten by command line argument
#  get current file path
_C.DATA.TRAIN_PATH = ""
# test data path
_C.DATA.TEST_PATH = ""
# validation data path
_C.DATA.VAL_PATH = ""
# img height
_C.DATA.CROP_IMG_HEIGHT = 224
# img width
_C.DATA.CROP_IMG_WIDTH = 224
# img norm mean
_C.DATA.IMG_NORM_MEAN = [0.485, 0.456, 0.406]
# img norm std
_C.DATA.IMG_NORM_STD = [0.229, 0.224, 0.225]
# num of workers
_C.DATA.NUM_WORKERS = 4


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = ''
_C.MODEL.NUM_VERTICES = 100
# model type
_C.MODEL.TYPE = ''
# Resume model path
_C.MODEL.RESUME = ""
# Pretrained model path
_C.MODEL.PRETRAINED = ""
# Dropout rate
_C.MODEL.DROPOUT = 0.0
_C.MODEL.D_MODEL = 1024
_C.MODEL.DEPTH = 20
_C.MODEL.NUM_HEADS = 16
_C.MODEL.PATCH_SIZE = 8
_C.MODEL.MLP_DIM = 8


# CrossFormer parameters
_C.MODEL.CROS = CN()
_C.MODEL.CROS.PATCH_SIZE = [4, 8, 16, 32]
_C.MODEL.CROS.MERGE_SIZE = [[2, 4], [2,4], [2, 4]]
_C.MODEL.CROS.IN_CHANS = 3
_C.MODEL.CROS.EMBED_DIM = 48
_C.MODEL.CROS.DEPTHS = [2, 2, 6, 2]
_C.MODEL.CROS.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.CROS.GROUP_SIZE = [7, 7, 7, 7]
_C.MODEL.CROS.MLP_RATIO = 4.
_C.MODEL.CROS.QKV_BIAS = True
_C.MODEL.CROS.QK_SCALE = None
_C.MODEL.CROS.APE = False
_C.MODEL.CROS.PATCH_NORM = True


# -----------------------------------------------------------------------------
# SMPL settings
# -----------------------------------------------------------------------------

_C.SMPL = CN()
# SMPL model path
_C.SMPL.SMPL_MODEL_DIR = ""
# SMPL mean shape path
_C.SMPL.SMPL_MEAN_PARAMS = ""


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.DISTRIBUTED = False


# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []
_C.TRAIN.LR_SCHEDULER.T_INIT = 80000
_C.TRAIN.LR_SCHEDULER.LR_MIN = 1e-5
_C.TRAIN.LR_SCHEDULER.WARMUP_T = 10000
_C.TRAIN.LR_SCHEDULER.WARMUP_LR_INIT = 1e-6
_C.TRAIN.LR_SCHEDULER.MAX_LR = 1e-4

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# guassian nosie
_C.AUG.GUASS_VAR = (0, 0.5)

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT_DIR = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Frequency to evaluate
_C.LOG_FREQ = 10
# Frequency to logging training info
_C.TRAIN_FREQ = 10
# Frequency to logging validation info
_C.VALID_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.RANK = 0
# World size for DistributedDataParallel, given by command line argument
_C.WORLD_SIZE = 1
# checlpoint path
_C.CKPT_DIR = ""
# template mesh path
_C.TEMPLATE_MESH = ""
_C.SCALE = 0
# experiment name
_C.EXP_NAME = ""
# cache path
_C.CACHE = "cache"


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    # if args.opts:
    #     config.merge_from_list(args.opts)

    # def _check_args(name):
    #     if hasattr(args, name) and eval(f'args.{name}'):
    #         return True
    #     return False

    # merge from specific arguments
    # if _check_args('WORLD_SIZE'):
    #     config.WORLD_SIZE = args.batch_size
    # if _check_args('data_path'):
    #     config.RANK = args.data_path

    config.freeze()

def get_config(opt):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, opt)

    return config
