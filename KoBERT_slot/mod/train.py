# -*- coding: utf-8 -*-

import os
import pickle
import argparse

from utils import Reader
from to_array.bert_to_array import BERTToArray
from to_array.tags_to_array import TagsToArray
from models.bert_slot_model import BertSlotModel

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


if __name__ == "__main__":
    # reads command-line parameters
    parser = argparse.ArgumentParser("Training the BERT NLU model")
    parser.add_argument("--train", "-t",
                        help="Path to training set",
                        type=str,
                        required=True)
    parser.add_argument("--val", "-v",
                        help="Path to validation set",
                        type=str,
                        default="",
                        required=False)
    parser.add_argument("--save", "-s",
                        help="Folder path to save the trained model",
                        type=str,
                        required=True)
    parser.add_argument("--epochs", "-e",
                        help="Number of epochs",
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument("--batch", "-bs",
                        help="Batch size",
                        type=int,
                        default=64,
                        required=False)
    
    args = parser.parse_args()
    train_data_folder_path = args.train
    val_data_folder_path = args.val
    save_folder_path = args.save
    epochs = args.epochs
    batch_size = args.batch
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    tf.compat.v1.random.set_random_seed(7)
    config = tf.ConfigProto(intra_op_parallelism_threads=0, 
                            inter_op_parallelism_threads=0,
                            allow_soft_placement=True,
                            device_count = {"GPU": 1})
    sess = tf.compat.v1.Session(config=config)
    
    bert_model_hub_path = "/content/drive/MyDrive/Slot_tagging_project/code/bert-module"
    bert_vocab_path = os.path.join(bert_model_hub_path,
                                   "assets/vocab.korean.rawtext.list")

    print("reading training set")
    train_text_arr, train_tags_arr = Reader.read(train_data_folder_path)
    
    print("train_text_arr[0:2] :", train_text_arr[0:2])
    print("train_tags_arr[0:2] :", train_tags_arr[0:2])
    
    bert_to_array = BERTToArray(bert_vocab_path) 
    # bert_to_array MUST NOT tokenize input !!!
    
    t_input_ids, t_input_mask, t_segment_ids = bert_to_array.transform(train_text_arr)
    
    tags_to_array = TagsToArray()
    tags_to_array.fit(train_tags_arr)
    train_tags = tags_to_array.transform(train_tags_arr, t_input_ids)
    print("train_tags :", train_tags[0:2])
    slots_num = len(tags_to_array.label_encoder.classes_)
    print("slot num :", slots_num, tags_to_array.label_encoder.classes_)
    
    model = BertSlotModel(slots_num, bert_model_hub_path, sess,
                        num_bert_fine_tune_layers=10)
    
    print("train input shape :", t_input_ids.shape, t_input_ids[0:2])
    print("t_input_mask :", t_input_mask.shape, t_input_mask[0:2])
    print("t_segment_ids :", t_segment_ids.shape, t_segment_ids[0:2])
    print("train_tags :", train_tags.shape, train_tags[0:2])
    
    if val_data_folder_path:
        print("reading validation set")

        val_text_arr, val_tags_arr = Reader.read(val_data_folder_path)
        v_input_ids, v_input_mask, v_segment_ids = bert_to_array.transform(val_text_arr)
    
        tags_to_array.fit(val_tags_arr)
        val_tags = tags_to_array.transform(val_tags_arr, v_input_ids)
        
        print("training model ...")
        model.fit([t_input_ids, t_input_mask, t_segment_ids],
                  train_tags,
                  validation_data=([v_input_ids, v_input_mask, v_segment_ids], val_tags),
                  epochs=epochs, batch_size=batch_size)

    else:
        print("training model ...")
        model.fit([t_input_ids, t_input_mask, t_segment_ids],
                  train_tags,
                  validation_data=None,
                  epochs=epochs, batch_size=batch_size)
    
    ### saving
    print("Saving ..")
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
        print("Folder `%s` created" % save_folder_path)
    model.save(save_folder_path)
    tags_to_array_path = os.path.join(save_folder_path, "tags_to_array.pkl")
    with open(tags_to_array_path, "wb") as handle:
        pickle.dump(tags_to_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    tf.compat.v1.reset_default_graph()
