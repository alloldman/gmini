import tensorflow as tf
import numpy as np
import csv
import os
import sys
import time
import networkx as nx
import itertools
from tqdm import tqdm

# sys.path.append('/home/simon/zpl/FuncSim')
sys.path.append('/home/xiqian/Binary_similarity/FuncSim/FuncSim')

import core.config as config
from core.config import logging
from core.tfrecord_core import load_dataset, generate_cfg_pair
from core.tfrecord_core import generate_feature_pair


def construct_learning_dataset_gemini(pair_list):
    cfgs_1, cfgs_2 = generate_cfg_pair(pair_list)
    feas_1, feas_2, max_size, nums_1, nums_2 = \
                                        generate_feature_pair(pair_list, 0)
    return cfgs_1, cfgs_2, feas_1, feas_2, nums_1, nums_2, max_size


def gen_tfrecord_and_save_gemini(save_file, pair_list, label_list):
    cfgs_1, cfgs_2, feas_1, feas_2, nums_1, nums_2, max_size = \
                                construct_learning_dataset_gemini(pair_list)
    node_list = np.linspace(max_size, max_size, len(label_list), dtype=int)
    writer = tf.python_io.TFRecordWriter(save_file)
    logging.info('generate tfrecord and save to {}'.format(save_file))
    for item1, item2, item3, item4, item5, item6, item7, item8 in tqdm(zip( \
            label_list, cfgs_1, cfgs_2, feas_1, feas_2, nums_1, nums_2, node_list)):
        feature = {
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[item1])),
            'cfg_1': tf.train.Feature(bytes_list = tf.train.BytesList(value=[item2])),
            'cfg_2': tf.train.Feature(bytes_list = tf.train.BytesList(value=[item3])),
            'fea_1': tf.train.Feature(bytes_list = tf.train.BytesList(value=[item4])),
            'fea_2': tf.train.Feature(bytes_list = tf.train.BytesList(value=[item5])),
            'num_1': tf.train.Feature(int64_list = tf.train.Int64List(value=[item6])),
            'num_2': tf.train.Feature(int64_list = tf.train.Int64List(value=[item7])),
            'max': tf.train.Feature(int64_list = tf.train.Int64List(value=[item8])),
        }
        features = tf.train.Features(feature = feature)
        example_proto = tf.train.Example(features = features)
        serialized = example_proto.SerializeToString()
        writer.write(serialized)
    writer.close()


train_pair, train_label, valid_pair, valid_label, test_pair, test_label = load_dataset()
logging.info('generate tfrecord: gemini train...')
gen_tfrecord_and_save_gemini(config.TFRECORD_GEMINI_TRAIN, train_pair, train_label)
logging.info('generate tfrecord: gemini valid...')
gen_tfrecord_and_save_gemini(config.TFRECORD_GEMINI_VALID, valid_pair, valid_label)
logging.info('generate tfrecord: gemini test...')
gen_tfrecord_and_save_gemini(config.TFRECORD_GEMINI_TEST, test_pair, test_label)

