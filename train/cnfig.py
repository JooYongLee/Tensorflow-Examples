import numpy as np
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by:
cnfig = __C
__TRAIN = edict()
__TEST = edict()

MODE_KEY={
    "TRAIN":0,
    "TEST":1
}

TRAIN = __TRAIN
TEST = __TEST
IMAGE_WIDTH = 107
IMAGE_HEIGHT = 107


NUM_CALSS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 50
TRAIN.EPOCH = 100