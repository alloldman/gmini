import os
import sys
import logging

sys.path.append(os.path.dirname(__file__))

logging.basicConfig(stream = sys.stdout, \
                    # filename='logs/gen_train_dataset.log', \
                    # level=logging.DEBUG)
                    level=logging.INFO)

PROGRAMS = ['openssl', 'busybox']
ARCHS = ['all']# ['aarch64', 'powerpc64'] # ['64']
OPT_LEVELS = ['all'] # ['o0', 'o1', 'o2', 'o3']
TRAIN_DATASET_NUM = 100000

WORD2VEC_EMBEDDING_SIZE = 50

# TRAIN
NUM_EPOCH = 5

# DATA ROOT
DATA_ROOT_DIR = '/home/xiqian/Binary_similarity/FuncSim/data'

# FEATURE
FEA_DIR = os.path.join(DATA_ROOT_DIR, 'features')
CFG_DFG_GEMINIFEA_VULSEEKERFEA = 'cfg_dfg_geminifea_vulseekerfea'
I2VFEA = 'i2v_norm2_' + str(WORD2VEC_EMBEDDING_SIZE)

FEATURE_GEMINI_DIMENSION = 7
FEATURE_VULSEEKER_DIMENSION = 8
FEATURE_I2V_DIMENSION = WORD2VEC_EMBEDDING_SIZE

# DATASET
DATASET_MIN_BLOCK_NUM = 5
DATASET_MAX_BLOCK_NUM = 30

FILENAME_PREFIX = str(TRAIN_DATASET_NUM) + '_[' + \
                str(DATASET_MIN_BLOCK_NUM) + '_' + \
                str(DATASET_MAX_BLOCK_NUM) + ']_[' + \
                '_'.join(PROGRAMS) + ']_[' + \
                '_'.join(ARCHS) + ']_[' + \
                '_'.join(OPT_LEVELS) + ']'

DATASET_DIR = os.path.join(DATA_ROOT_DIR, 'datasets')
DATASET_TRAIN = os.path.join(DATASET_DIR, 'train' + FILENAME_PREFIX + '.csv')
DATASET_VALID = os.path.join(DATASET_DIR, 'valid' + FILENAME_PREFIX + '.csv')
DATASET_TEST = os.path.join(DATASET_DIR, 'test' + FILENAME_PREFIX + '.csv')

# TFRECORD
TFRECORD_DIR = os.path.join(DATA_ROOT_DIR, 'tfrecords')

GEN_TFRECORD_GEMINI = True
TFRECORD_GEMINI_DIR = os.path.join(TFRECORD_DIR, 'gemini')

TFRECORD_GEMINI_TRAIN = os.path.join(TFRECORD_GEMINI_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_GEMINI_VALID = os.path.join(TFRECORD_GEMINI_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_GEMINI_TEST = os.path.join(TFRECORD_GEMINI_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_I2V_GEMINI = True
TFRECORD_I2V_GEMINI_DIR = os.path.join(TFRECORD_DIR, 'i2v_gemini')

TFRECORD_I2V_GEMINI_TRAIN = os.path.join(TFRECORD_I2V_GEMINI_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_GEMINI_VALID = os.path.join(TFRECORD_I2V_GEMINI_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_GEMINI_TEST = os.path.join(TFRECORD_I2V_GEMINI_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_VULSEEKER = True
TFRECORD_VULSEEKER_DIR = os.path.join(TFRECORD_DIR, 'vulseeker')

TFRECORD_VULSEEKER_TRAIN = os.path.join(TFRECORD_VULSEEKER_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_VULSEEKER_VALID = os.path.join(TFRECORD_VULSEEKER_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_VULSEEKER_TEST = os.path.join(TFRECORD_VULSEEKER_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_I2V_VULSEEKER = True
TFRECORD_I2V_VULSEEKER_DIR = os.path.join(TFRECORD_DIR, 'i2v_vulseeker')

TFRECORD_I2V_VULSEEKER_TRAIN = os.path.join(TFRECORD_I2V_VULSEEKER_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_VULSEEKER_VALID = os.path.join(TFRECORD_I2V_VULSEEKER_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_VULSEEKER_TEST = os.path.join(TFRECORD_I2V_VULSEEKER_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_FUNCSIM = True
TFRECORD_FUNCSIM_DIR = os.path.join(TFRECORD_DIR, 'funcsim')

TFRECORD_FUNCSIM_TRAIN = os.path.join(TFRECORD_FUNCSIM_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_FUNCSIM_VALID = os.path.join(TFRECORD_FUNCSIM_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_FUNCSIM_TEST = os.path.join(TFRECORD_FUNCSIM_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

GEN_TFRECORD_I2V_FUNCSIM = True
TFRECORD_I2V_FUNCSIM_DIR = os.path.join(TFRECORD_DIR, 'i2v_funcsim')

TFRECORD_I2V_FUNCSIM_TRAIN = os.path.join(TFRECORD_I2V_FUNCSIM_DIR, \
                                'train' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_FUNCSIM_VALID = os.path.join(TFRECORD_I2V_FUNCSIM_DIR, \
                                'valid' + FILENAME_PREFIX + '.tfrecord')
TFRECORD_I2V_FUNCSIM_TEST = os.path.join(TFRECORD_I2V_FUNCSIM_DIR, \
                                'test' + FILENAME_PREFIX + '.tfrecord')

# MODEL
MODEL_DIR = os.path.join(DATA_ROOT_DIR, 'models')

MODEL_GEMINI_DIR = os.path.join(MODEL_DIR, 'gemini')

MODEL_I2V_GEMINI_DIR = os.path.join(MODEL_DIR, 'i2v_gemini')

MODEL_VULSEEKER_DIR = os.path.join(MODEL_DIR, 'vulseeker')

MODEL_I2V_VULSEEKER_DIR = os.path.join(MODEL_DIR, 'i2v_vulseeker')

MODEL_FUNCSIM_DIR = os.path.join(MODEL_DIR, 'funcsim')

MODEL_I2V_FUNCSIM_DIR = os.path.join(MODEL_DIR, 'i2v_funcsim')

# STATIS
STATIS_DIR = os.path.join(DATA_ROOT_DIR, 'statis')

STATIS_GEMINI_DIR = os.path.join(STATIS_DIR, 'gemini')

STATIS_I2V_GEMINI_DIR = os.path.join(STATIS_DIR, 'i2v_gemini')

STATIS_VULSEEKER_DIR = os.path.join(STATIS_DIR, 'vulseeker')

STATIS_I2V_VULSEEKER_DIR = os.path.join(STATIS_DIR, 'i2v_vulseeker')

STATIS_FUNCSIM_DIR = os.path.join(STATIS_DIR, 'funcsim')

STATIS_I2V_FUNCSIM_DIR = os.path.join(STATIS_DIR, 'i2v_funcsim')

def config_test_and_create_dirs(*args):
    for fname in args:
        d = fname
        if os.path.isfile(fname):
            d, f = os.path.split(fname)
        if not os.path.exists(d):
            os.makedirs(d)

config_test_and_create_dirs( \
        DATA_ROOT_DIR, \
        FEA_DIR, \
        DATASET_DIR, \
        TFRECORD_DIR, \
        TFRECORD_GEMINI_DIR, \
        TFRECORD_I2V_GEMINI_DIR, \
        TFRECORD_VULSEEKER_DIR, \
        TFRECORD_I2V_VULSEEKER_DIR, \
        TFRECORD_FUNCSIM_DIR, \
        TFRECORD_I2V_FUNCSIM_DIR, \
        MODEL_DIR, \
        MODEL_GEMINI_DIR, \
        MODEL_I2V_GEMINI_DIR, \
        MODEL_VULSEEKER_DIR, \
        MODEL_I2V_VULSEEKER_DIR, \
        MODEL_FUNCSIM_DIR, \
        MODEL_I2V_FUNCSIM_DIR, \
        STATIS_DIR, \
        STATIS_GEMINI_DIR, \
        STATIS_I2V_GEMINI_DIR, \
        STATIS_VULSEEKER_DIR, \
        STATIS_I2V_VULSEEKER_DIR, \
        STATIS_FUNCSIM_DIR, \
        STATIS_I2V_FUNCSIM_DIR, \
    )
