# -*- coding: utf-8 -*-

from itertools import chain
import os
import spliter

def flatten(y):
    return list(chain.from_iterable(y))

class Reader:
    
    def __init__(self):
        pass
    
    def read(dataset_folder_path):
        text_arr = []
        text_ap = text_arr.append
        tags_arr = []
        tags_ap = tags_arr.append
        
        ########################### TODO ###############################
        # seq.in과 seq.out을 읽어들여서 리스트로 만들기
        # (텍스트 파일의 한 라인이 리스트의 한 요소가 되도록)
        with open(os.path.join(dataset_folder_path,'seq.in'),'r') as f:
          for sentence in f.readlines():
            text_arr.append(sentence)

        with open(os.path.join(dataset_folder_path,'seq.out'),'r') as f:
          for sentence in f.readlines():
            tags_arr.append(sentence)
        ################################################################

        assert len(text_arr) == len(tags_arr)
        return text_arr, tags_arr
