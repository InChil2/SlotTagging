# -*- coding: utf-8 -*-

import os
import pickle
import argparse


from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel


import tensorflow as tf


if __name__ == "__main__":
    # Reads command-line parameters
    parser = argparse.ArgumentParser("Evaluating the BERT NLU model")
    parser.add_argument("--model", "-m",
                        help="Path to BERT NLU model",
                        type=str,
                        required=True)
    
    args = parser.parse_args()
    load_folder_path = args.model
    

    # this line is to disable gpu
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1,
                            allow_soft_placement=True,
                            device_count = {"CPU": 1})
    sess = tf.compat.v1.Session(config=config)


    bert_model_hub_path = "/content/drive/MyDrive/Slot_tagging_project/code/bert-module"
    vocab_file = os.path.join(bert_model_hub_path, 'assets/vocab.korean.rawtext.list')
    bert_to_array = BERTToArray(vocab_file)

    print('Loading Models...')
    if not os.path.exists(load_folder_path):
        print(f'Folder {load_folder_path} not exist')

    tags_to_array_path = os.path.join(load_folder_path, 'tags_to_array.pkl')
    with open(tags_to_array_path, 'rb') as handle:
        tags_to_array = pickle.load(handle)
        slots_num = len(tags_to_array.label_encoder.classes_)

    model = BertSlotModel.load(load_folder_path, sess)

    while True:
        print('Enter Your Sentence : ')

        try :
            input_text = input().strip()

        except :
            continue

        if input_text in ['quit', '종료', '그만', '멈춰', 'stop']:
            break

        else :
            text_arr = bert_to_array.tokenizer.tokenize(input_text)

            input_ids, input_mask, segment_ids = bert_to_array.transform([' '.join(text_arr)])

            inferred_tags, slot_score = model.predict_slots([input_ids, input_mask, segment_ids], tags_to_array)

            print(text_arr)
            print(inferred_tags[0])
            print(slot_score[0])
    
    tf.compat.v1.reset_default_graph()
    
