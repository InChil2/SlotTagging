import pandas as pd
import os
import numpy as np


def train_test_val_split():

    data_dir = '/content/Slot_tagging_project'

    in_li = []
    in_ap = in_li.append

    out_li = []
    out_ap = out_li.append

    train_in = []
    train_inap = train_in.append
    train_out = []
    train_outap = train_out.append

    test_in = []
    test_inap = test_in.append
    test_out = []
    test_outap = test_out.append

    val_in = []
    val_inap = val_in.append
    val_out = []
    val_outap = val_out.append



    with open(os.join.path(data_dir,'/data/result_data/seq.in'),'r') as f:
        for sentence in f.readlines():
            in_ap(sentence.strip())

    with open(os.join.path(data_dir,'/data/result_data/seq.out'),'r') as f:
        for sentence in f.readlines():
            out_ap(sentence.strip())

    num_li = list(range(len(in_li)))
    random.shuffle(num_li)

    val = len(num_li) // 10
    test = len(num_li) // 10
    train = len(num_li) - (val + test)

    val_idx = num_li[:val]
    test_idx = num_li[val:val+test]
    train_idx = num_li[val+test:]

    for i in val_idx:
        val_inap(in_li[i])
        val_outap(out_li[i])

    for i in test_idx:
        test_inap(in_li[i])
        test_outap(out_li[i])

    for i in train_idx:
        train_inap(in_li[i])
        train_outap(out_li[i])

    with open(os.join.path(data_dir,'data/datasets/test/seq.in'),'w') as f:
        for line in test_in:
            f.write(line+'\n')

    with open(os.join.path(data_dir,'data/datasets/test/seq.out'),'w') as f:
        for line in test_out:
            f.write(line+'\n')

    with open(os.join.path(data_dir,'data/datasets/val/seq.in'),'w') as f:
        for line in val_in:
            f.write(line+'\n')

    with open(os.join.path(data_dir,'data/datasets/val/seq.out'),'w') as f:
        for line in val_out:
            f.write(line+'\n')

    with open(os.join.path(data_dir,'data/datasets/train/seq.in'),'w') as f:
        for line in train_in:
            f.write(line+'\n')

    with open(os.join.path(data_dir,'data/datasets/train/seq.out'),'w') as f:
        for line in test_out:
            f.write(line+'\n')

if __name__ == "__main__":
    train_test_val_split()