from torch.optim.sgd import SGD

BATCH_SIZE = 300
LR = 1e-1
MOMENTUM = 0
EPOCHS = 30
SAVE_INTERVAL = 1
EVAL_INTERVAL = 1
OPTIMIZER = SGD #SGD
INPUT_SIZE = (32, 32)
# DATASET = 'imagenet'
# DATASET_ROOT = '/'
DATASET = 'mnist'
DATASET_ROOT = './data'
