# -*- coding: utf-8 -*-

from itertools import chain
import os

def flatten(y):
    return list(chain.from_iterable(y))

class Reader:
    def __init__(self):
        pass
    def read(dataset_folder_path):
        text_arr = []
        tags_arr = []

        in_ap = text_arr.append
        out_ap = tags_arr.append

        with open(os.path.join(dataset_folder_path,'seq.in'),'r') as f:
          for sentence in f.readlines():
            in_ap(sentence.strip())

        with open(os.path.join(dataset_folder_path,'seq.out'),'r') as f:
          for sentence in f.readlines():
            out_ap(sentence.strip())

        assert len(text_arr) == len(tags_arr)
        return text_arr, tags_arr
    