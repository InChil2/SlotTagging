# -*- coding: utf-8 -*-

import os, re
import argparse
from tokenizationK import FullTokenizer

import pdb

############################################## TODO 경로 고치기 ###############################################
tokenizer = FullTokenizer(vocab_file="/content/drive/MyDrive/Slot_tagging_project/code/bert-module/assets/vocab.korean.rawtext.list")
###############################################################################################################

# "/인물;한지민/과 /인물;한예슬/ 나오는 드라마 있어?"와 같은 예시처럼
# 해당 데이터에서는 "/슬롯(레이블)명;엔티티/"의 형식으로 슬롯과 엔티티를 정리해 놨으므로,
# 이를 잡아 줄 수 있는 정규표현식을 준비한다.
slot_pattern = re.compile(r"/(.+?);(.+?)/")
multi_spaces = re.compile(r"\s+")


def process_file(file_path, output_dir):
    """
    단방향 데이터가 있는 file_path을 argument로 주면 가공을 한 이후에
    output_dir 아래에 2개의 파일(seq.in, seq.out)을 저장해 주는 함수.
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    data = open(file_path).read().splitlines()

    # line별로 process를 해준 뒤,
    processed_data = [process_line(line, tokenizer) for line in data]

    tokens = list(map(lambda x: x[0], processed_data))
    tags = list(map(lambda x: x[1], processed_data))

    # seq_in : 토큰들로만 이루어진 파일
    # seq_out : 태그들로만 이루어진 파일
    seq_in = os.path.join(output_dir, "seq.in")
    seq_out = os.path.join(output_dir, "seq.out")

    with open(seq_in, "w") as f:
        f.write("\n".join(tokens)+ "\n")

    with open(seq_out, "w") as f:
        f.write("\n".join(tags)+ "\n")


def process_line(sentence, tokenizer):
    """
    데이터를 라인별로 처리해 주는 함수이다.
    라인을 주게 되면, (토큰, 슬롯)

    예를 들어 "/인물;한지민/과 /인물;한예슬/ 나오는 드라마 있어?" 같은 input을 받게 되면,
        ('한 지민 과 한예 슬 나오 는 드라마 있 어 ?', '인물 인물 O 인물 인물 O O O O O O')와 같은 (토큰, 태그)쌍으로 된 결과값을 반환한다.
    """
    slot_pattern_found = slot_pattern.findall(sentence)
    line_refined = slot_pattern.sub("/슬롯/", sentence)
    tokens = ""
    tags = ""
    slot_index = 0

    for word in line_refined.split():
        # "/게임명;일곱개의 대죄/" --> ("게임명", "일곱개의 대죄")
        if word.startswith("/"):
            slot, entity = slot_pattern_found[slot_index]
            slot_index += 1

            # 엔티티를 토크나이즈 한 후, 토큰별로 태그를 추가해 준다.
            entity_tokens = " ".join(tokenizer.tokenize(entity))

            tokens += entity_tokens + " "
            tags += (slot + " ") * len(entity_tokens.split())

            # 조사가 붙은 것이며(eg. "/슬롯/이", "/슬롯/에서"),
            # 조사에 대해서 추가적으로 토큰 및 태그를 추가해 준다.
            if not word.endswith("/"):
                # 우선 "/" 뒤에 오는 조사를 찾아 주고
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

    # 만일 토큰의 개수와 슬롯의 개수가 맞지 않다면 본래 라인과 더불어 토큰/슬롯들을 프린트해준다.
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
