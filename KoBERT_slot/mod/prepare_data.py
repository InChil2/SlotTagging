# -*- coding: utf-8 -*-

import os, re
import argparse
from tokenizationK import FullTokenizer
import pdb

tokenizer = FullTokenizer(vocab_file="/content/drive/MyDrive/Slot_tagging_project/code/bert-module/assets/vocab.korean.rawtext.list")

slot_pattern = re.compile(r"/(.+?);(.+?)/")
multi_spaces = re.compile(r"\s+")

def process_file(file_path, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    data = open(file_path).read().splitlines()

    processed_data = [process_line(line, tokenizer) for line in data]

    tokens = list(map(lambda x: x[0], processed_data))
    tags = list(map(lambda x: x[1], processed_data))

    seq_in = os.path.join(output_dir, "seq.in")
    seq_out = os.path.join(output_dir, "seq.out")

    with open(seq_in, "w") as f:
        f.write("\n".join(tokens)+ "\n")

    with open(seq_out, "w") as f:
        f.write("\n".join(tags)+ "\n")


def process_line(sentence, tokenizer):
    slot_pattern_found = slot_pattern.findall(sentence)
    line_refined = slot_pattern.sub("/슬롯/", sentence)
    tokens = ""
    tags = ""
    slot_index = 0

    for word in line_refined.split():
        if word.startswith("/"):
            slot, entity = slot_pattern_found[slot_index]
            slot_index += 1

            entity_tokens = " ".join(tokenizer.tokenize(entity))

            tokens += entity_tokens + " "
            tags += (slot + " ") * len(entity_tokens.split())

            if not word.endswith("/"):
                josa = word[word.rfind("/")+1:]
                josa_tokens = " ".join(tokenizer.tokenize(josa))

                tokens += josa_tokens + " "
                tags += "O " * len(josa_tokens.split())
            
        elif "/" in word:

            prefix = word.split("/")[0]
            tokenized_prefix = " ".join(tokenizer.tokenize(prefix))
            tokens += tokenized_prefix + " "
            tags += "O " * len(tokenized_prefix.split())

            slot, entity = slot_pattern_found[slot_index]
            slot_index += 1

            entity_tokens = " ".join(tokenizer.tokenize(entity))

            tokens += entity_tokens + " "
            tags += (slot + " ") * len(entity_tokens.split())

        else:
            word_tokens = " ".join(tokenizer.tokenize(word))

            tokens += word_tokens + " "
            tags += "O " * len(word_tokens.split())

    tokens = multi_spaces.sub(" ", tokens.strip())
    tags = multi_spaces.sub(" ", tags.strip())

    if len(tokens.split()) != len(tags.split()):
        print(sentence)
        print("\t" + tokens + "\t", len(tokens.split()))
        print("\t" + tags + "\t", len(tags.split()))

    return tokens, tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help = '단방향 데이터 형식의 텍스트 파일', type = str, required = True)
    parser.add_argument('--output', '-o', help = 'Path to data', type = str, required = True)

    args = parser.parse_args()
    file_path = args.input
    output_dir = args.output

    process_file(file_path, output_dir)
    
