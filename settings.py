import logging
import time
import os.path as osp

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False
# EVAL = True
# dataset can be 'WIKI', 'MIRFlickr' or 'NUSWIDE'
# DATASET = 'NUSWIDE'
DATASET = 'MIRFlickr'
# DATASET = 'MSCOCO'
# DATASET = 'WIKI'
# DATASET = 'MIRFlickr'
if DATASET == 'WIKI':

    DATA_DIR = './datasets/WIKI/images'
    LABEL_DIR = './datasets/WIKI/raw_features.mat'
    TRAIN_LABEL = './datasets/WIKI/trainset_txt_img_cat.list'
    TEST_LABEL = './datasets/WIKI/testset_txt_img_cat.list'

    BETA = 0.3
    LAMBDA1 = 0.3
    LAMBDA2 = 0.3
    NUM_EPOCH = 400
    LR_IMG = 0.01
    LR_TXT = 0.01
    EVAL_INTERVAL = 1


if DATASET == 'MIRFlickr':

    LABEL_DIR = './datasets/mirflickr/mirflickr25k-lall.mat'
    TXT_DIR = './datasets/mirflickr/mirflickr25k-yall.mat'
    IMG_DIR = './datasets/mirflickr/mirflickr25k-iall.mat'
    
    BETA = 0.9
    LAMBDA1 = 0.1
    LAMBDA2 = 0.1
    NUM_EPOCH = 100000
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 100

    alpha = 2.0
    beta = 4.0
    # threshold = 0.27
    # threshold = 0.05
    threshold = 0.200

    # threshold = 0.15
    K = 5
    
    

if DATASET == 'NUSWIDE':

    LABEL_DIR = './datasets/NUSWIDE/nus-wide-tc10-lall.mat'
    TXT_DIR = './datasets/NUSWIDE/nus-wide-tc10-yall.mat'
    IMG_DIR = './datasets/NUSWIDE/IAll/nus-wide-tc10-iall.mat'

    BETA = 0.6
    LAMBDA1 = 0.3
    LAMBDA2 = 0.3
    NUM_EPOCH = 70
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 2

    # threshold = -1
    threshold = -0.5

    K = 3

if DATASET == 'MSCOCO':
    LABEL_DIR = './data_pre/output/cocoLAll-8w.mat'
    TXT_DIR = './data_pre/output/cocoYAll-8w.mat'
    IMG_DIR = './data_pre/output/cocoIAll-8w.mat'
    
    BETA = 0.6
    LAMBDA1 = 0.3
    LAMBDA2 = 0.3
    NUM_EPOCH = 200
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 1

    # threshold = -1
    threshold = 0.5
    K = 3




BATCH_SIZE = 128
CODE_LEN = 16
l4 = 0.3
l5 = 0.0005
sim1 = 1.4
nnk = 0.08
temperature =  1.0

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
GPU_ID = 0
NUM_WORKERS = 8
EPOCH_INTERVAL = 2

MODEL_DIR = './HCAC/checkpoint'


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())) 
log_name = now + '_log.txt'
log_dir = './HCAC/log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

