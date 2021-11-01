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
        
        ########################### TODO ###############################
        # seq.in과 seq.out을 읽어들여서 리스트로 만들기
        # (텍스트 파일의 한 라인이 리스트의 한 요소가 되도록)
        f_seq_in = open('/content/drive/MyDrive/Slot_tagging_project/code/박민아/test/seq.in', 'r')
        f_seq_out = open('/content/drive/MyDrive/Slot_tagging_project/code/박민아/test/seq.out', 'r')

        for line in f_seq_in.readlines():
          line = line.strip()
          text_arr.append(line)
        f_seq_in.close()

        for line in f_seq_out.readlines():
          line = line.strip()
          tags_arr.append(line)
        f_seq_out.close()
        ################################################################

        assert len(text_arr) == len(tags_arr)
        return text_arr, tags_arr