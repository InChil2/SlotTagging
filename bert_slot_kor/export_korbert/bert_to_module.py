import re
import os
import sys
import json
import logging
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from modeling import BertModel, BertConfig


# get TF logger 
log = logging.getLogger("tensorflow")
log.handlers = []

def build_module_fn(config_path, vocab_path, do_lower_case=True):

    def bert_module_fn(is_training):
        """Spec function for a token embedding module."""

        input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
        input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
        token_type = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")

        config = BertConfig.from_json_file(config_path)
        model = BertModel(config=config, is_training=is_training,
                          input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type)
          
        model.input_to_output()
        seq_output = model.get_all_encoder_layers()[-1]

        config_file = tf.constant(value=config_path, dtype=tf.string, name="config_file")
        vocab_file = tf.constant(value=vocab_path, dtype=tf.string, name="vocab_file")
        lower_case = tf.constant(do_lower_case)

        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, config_file)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_file)
        
        input_map = {"input_ids": input_ids,
                     "input_mask": input_mask,
                     "segment_ids": token_type}
        
        output_map = {"sequence_output": seq_output}

        output_info_map = {"vocab_file": vocab_file,
                           "do_lower_case": lower_case}
                
        hub.add_signature(name="tokens", inputs=input_map, outputs=output_map)
        hub.add_signature(name="tokenization_info", inputs={}, outputs=output_info_map)

    return bert_module_fn


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", help = "Etri KorBERT dir", type = str, required=True)
    parser.add_argument("--output_dir", "-o", help = "output_dir", type = str, required=True)

    args = parser.parse_args()
    MODEL_DIR = args.input_dir
    output_dir = args.output_dir

    config_path = "{}/bert_config.json".format(MODEL_DIR)
    vocab_path = "{}/vocab.korean.rawtext.list".format(MODEL_DIR)

    tags_and_args = []
    for is_training in (True, False):
      tags = set()
      if is_training:
        tags.add("train")
      tags_and_args.append((tags, dict(is_training=is_training)))

    module_fn = build_module_fn(config_path, vocab_path)
    spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args)
    spec.export(output_dir, 
                checkpoint_path="{}/model.ckpt-56000".format(MODEL_DIR))
    print("complete")


