# -*- coding: utf-8 -*-

import sys

import numpy as np
import tensorflow as tf

from to_array import tokenizationK as tk


class BERTToArray:
    
    def __init__(self
                 bert_vocab_path="./bert-module/assets/vocab.korean.rawtext.list"):
        self.tokenizer = tk.FullTokenizer(bert_vocab_path)
    
    def transform(self, text_arr):
        print('transform started')
        input_ids = []
        input_mask = []
        segment_ids = []

        for text in text_arr: 
            ids, mask, seg_ids= self.__to_array(text.strip())

            input_ids.append(ids)
            input_mask.append(mask)
            segment_ids.append(seg_ids)

        input_ids = tf.keras.preprocessing.sequence.pad_sequences(
            input_ids,
            padding='post'
        )
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(
            input_mask,
            padding='post'
        )
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(
            segment_ids,
            padding='post'
        )
        return input_ids, input_mask, segment_ids
    
    def __to_array(self, text: str):
        # str -> tokens list
        tokens = text.split() # whitespace tokenizer

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        
        return input_ids, input_mask, segment_ids
        
