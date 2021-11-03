import random
import os
from prepare_data import process_file

def data_to_seq():
  data_dir = '/content/drive/MyDrive/Slot_tagging_project'

  data_li = []
  data_ap = data_li.append

  with open(os.path.join(data_dir,'data/data.txt'),'r') as f:
    for i in f.readlines():
      data_ap(i.strip())

  random.shuffle(data_li)

  val = len(data_li) // 10
  test = len(data_li) // 10

  val_data = data_li[:val]
  test_data = data_li[val:val+test]
  train_data = data_li[val+test:]

  with open(os.path.join(data_dir,'data/datasets/val/data.txt'),'w') as f:
    for line in val_data:
      f.write(line+'\n')

  with open(os.path.join(data_dir,'data/datasets/test/data.txt'),'w') as f:
    for line in test_data:
      f.write(line+'\n')

  with open(os.path.join(data_dir,'data/datasets/train/data.txt'),'w') as f:
    for line in train_data:
      f.write(line+'\n')


  process_file('/content/drive/MyDrive/Slot_tagging_project/data/datasets/val/data.txt', '/content/drive/MyDrive/Slot_tagging_project/data/datasets/val')
  process_file('/content/drive/MyDrive/Slot_tagging_project/data/datasets/test/data.txt', '/content/drive/MyDrive/Slot_tagging_project/data/datasets/test')
  process_file('/content/drive/MyDrive/Slot_tagging_project/data/datasets/train/data.txt', '/content/drive/MyDrive/Slot_tagging_project/data/datasets/train')

if __name__ == '__main__':
    data_to_seq()