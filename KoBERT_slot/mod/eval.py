# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import tensorflow as tf
from sklearn import metrics
from utils import Reader
from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from utils import flatten

def get_results(input_ids, input_mask, segment_ids, tags_arr, tags_to_array):
    inferred_tags, slots_score = model.predict_slots([input_ids,
                                                        input_mask,
                                                        segment_ids],
                                                        tags_to_array)
    gold_tags = [x.split() for x in tags_arr]

    f1_score = metrics.f1_score(flatten(gold_tags), flatten(inferred_tags),
                                average="micro")

    tag_incorrect = ""
    for i, sent in enumerate(input_ids):
        if inferred_tags[i] != gold_tags[i]:
            tokens = bert_to_array.tokenizer.convert_ids_to_tokens(input_ids[i])
            tag_incorrect += "sent {}\n".format(tokens)
            tag_incorrect += ("pred: {}\n".format(inferred_tags[i]))
            tag_incorrect += ("score: {}\n".format(slots_score[i]))
            tag_incorrect += ("ansr: {}\n\n".format(gold_tags[i]))

    return f1_score, tag_incorrect

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluating the BERT NLU model")
    parser.add_argument("--model", "-m",
                        help="Path to BERT NLU model",
                        type=str,
                        required=True)
    parser.add_argument("--data", "-d",
                        help="Path to data",
                        type=str,
                        required=True)
    
    args = parser.parse_args()
    load_folder_path = args.model
    data_folder_path = args.data
    
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    
    config = tf.ConfigProto(intra_op_parallelism_threads=8, 
                            inter_op_parallelism_threads=0,
                            allow_soft_placement=True,
                            device_count = {"CPU": 8})
    sess = tf.Session(config=config)
    

    bert_model_hub_path = "/content/drive/MyDrive/Slot_tagging_project/code/bert-module"
    vocab_file = os.path.join(bert_model_hub_path,
                              "assets/vocab.korean.rawtext.list")
    bert_to_array = BERTToArray(vocab_file)
    
    print("Loading models ...")
    if not os.path.exists(load_folder_path):
        print("Folder `%s` not exist" % load_folder_path)
    
    tags_to_array_path = os.path.join(load_folder_path, "tags_to_array.pkl")
    with open(tags_to_array_path, "rb") as handle:
        tags_to_array = pickle.load(handle)
        slots_num = len(tags_to_array.label_encoder.classes_)
        
    model = BertSlotModel.load(load_folder_path, sess)

    print('reading test set')
    test_text_arr, test_tags_arr = Reader.read(data_folder_path)
    test_input_ids, test_input_mask, test_segment_ids = bert_to_array.transform(test_text_arr)

    f1_score, tag_incorrect = get_results(test_input_ids, test_input_mask,
                                          test_segment_ids, test_tags_arr,
                                          tags_to_array)
    
    result_path = os.path.join(load_folder_path, "test_results")
    
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    
    with open(os.path.join(result_path, "tag_incorrect.txt"), "w") as f:
        f.write(tag_incorrect)
    
    with open(os.path.join(result_path, "test_total.txt"), "w") as f:
        f.write("Slot f1_score = {}\n".format(f1_score))
    
    tf.compat.v1.reset_default_graph()
    print("complete")
    
