# -*- coding: utf-8 -*-
# main.py
# 작성 일 : 2021.11.07
# InCheol Shin
# Slot_Tagging_Project(web_demo)
# 수정 일 : 2021.11.09
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import tensorflow as tf
import os, pickle, re, sys
import pandas as pd
import datetime

# 필요한 모듈 불러오기
sys.path.append("/content/drive/MyDrive/Slot_tagging_project/code")
from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer

graph = tf.compat.v1.get_default_graph()

data_path = '/content/drive/MyDrive/Slot_tagging_project/data'
code_path = '/content/drive/MyDrive/Slot_tagging_project/code'

beverage = pd.read_csv(os.path.join(data_path,'beverage.csv'))['이름'].tolist()
food = pd.read_csv(os.path.join(data_path,'food.csv'))['이름'].tolist()
size = ['short','tall','venti','grande','숏','톨','벤티','그란데']
temperature = ['따뜻한', '뜨거운', 'hot', '차가운', '시원한','시원하게', 'ice', '아이스', 'cool', '따뜻하게', '차갑게']
quantity = ['한 잔','두 잔','세 잔','네 잔','다섯 잔','여섯 잔','일곱 잔','여덟 잔','아홉 잔','열 잔','하나','둘','셋','넷','다섯','여섯','일곱','여덟','아홉','1 잔','2 잔','3 잔','4 잔','5 잔','6 잔','7 잔','8 잔','9 잔','10 잔']
syrup = ['바닐라 시럽', '헤이즐넛 시럽', '카라멜 시럽', '클래식 시럽', '모카 시럽', '화이트 모카', '돌체 시럽']
syrup_quantity = ['한 번','두 번','세 번','네 번','다섯 번','여섯 번','일곱 번','여덟 번','아홉 번','열 번','한 펌프','두 펌프','세 펌프','다섯 펌프','여섯 펌프','일곱 펌프','여덟 펌프','아홉 펌프','열 펌프','1 번','2 번','3 번','4 번','5 번','6 번','7 번','8 번','9 번','10 번','1 펌프','2 펌프','3 펌프','4 펌프','5 펌프','6 펌프','7 펌프','8 펌프','9 펌프','10 펌프']
food_quantity = ['한 개','두 개','세 개','다섯 개','여섯 개','일곱 개','여덟 개','아홉 개','열 개','하나','둘','셋','넷','다섯','여섯','일곱','여덟','아홉','1 개','2 개','3 개','4 개','5 개','6 개','7 개','8 개','9 개','10 개']
members = ['이슬', '박민아', '신인철', '오수문', '정하림']
member_introduction = {'이슬':'이름 : 이슬</br>EMAIL : seuly1203@gmail.com</br>GIT HUB : github.com/seuly1203</br>역할 : 총괄',
'박민아':'이름 : 박민아</br>EMAIL : parkmina365@gmail.com</br>GIT HUB : github.com/parkmina365</br>역할 : 챗봇 알고리즘, BERT 모델링',
'정하림':'이름 : 정하림</br>EMAIL : halim7401@naver.com</br>GIT HUB : github.com/hharimjung</br>역할 : 웹 코딩, PPT',
'오수문':'이름 : 오수문</br>EMAIL : halim7401@naver.com</br>GIT HUB : github.com/sumunoh</br>역할 : 웹 코딩, 크롤링',
'신인철':'이름 : 신인철</br>EMAIL : snc4656@naver.com</br>GIT HUB : github.com/InChil2</br>역할 : 기술 총괄'}

menu = {'beverage':'음료',
        'food':'푸드',
        'size':'음료 사이즈',
        'temperature':'음료 온도',
        'quantity':'음료 수량',
        'syrup':'시럽',
        'syrup_quantity':'시럽 수량',
        'food_quantity':'푸드 수량',
        'members' : '팀원'}

dic = {i:globals()[i] for i in menu}

cmds = {'명령어':[],
        '음료':beverage,
        '푸드':food,
        '음료 사이즈':size,
        '음료 온도':temperature,
        '시럽':syrup,
        '팀원':members}

cmds["명령어"] = [k for k in cmds]

# enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    allow_soft_placement=True,
    device_count={"GPU": 0},
)
sess = tf.compat.v1.Session(config=config)


bert_model_hub_path = os.path.join(code_path,'bert-module')
load_folder_path = os.path.join(data_path,'save_models_ver1')

vocab_file = os.path.join(bert_model_hub_path, "assets/vocab.korean.rawtext.list")
bert_to_array = BERTToArray(vocab_file)

tags_to_array_path = os.path.join(load_folder_path, "tags_to_array.pkl")
with open(tags_to_array_path, "rb") as handle:
    tags_to_array = pickle.load(handle)
    slots_num = len(tags_to_array.label_encoder.classes_)

model = BertSlotModel.load(load_folder_path, sess)

tokenizer = FullTokenizer(vocab_file=vocab_file)

app = Flask(__name__)
run_with_ngrok(app) # 코랩 실행 시
app.static_folder = 'static'

@app.route("/dashboard", methods=['GET'])
def dashboard():
    path = os.path.join(data_path,'custinfo')
    file_list = os.listdir(path)
    
    so = len(file_list)
    oc = '3' # db 연결 필요
    warnal = '1' # db 연결 필요
    wa = '1' # db 연결 필요

    return render_template("dashboard.html", oc=oc, so=so, warnal=warnal, wa=wa)

@app.route("/message")
def message():
    return render_template("message.html")

@app.route("/orderlist")
def orderlist():
    return render_template("orderlist.html")

@app.route("/custlist")
def custlist():
    return render_template("custlist.html")

@app.route("/map")
def map():
    return render_template("map.html")

@app.route("/")
def home():
# 슬롯 사전 만들기
    app.slot_dict = {'beverage':[],
            'food':[],
            'size':[],
            'temperature':[],
            'quantity':[],
            'syrup':[],
            'syrup_quantity':[],
            'food_quantity':[]}

    app.score_limit = 0.6

    #변수 만들기

    return render_template("message.html")
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg').strip() # 사용자가 입력한 문장

    # if userText[0] == "!":
    #     try:
    #         li = cmds[userText[1:]]
    #         message = "<br />\n".join(li)
    #     except:
    #         message = "입력한 명령어가 존재하지 않습니다."

    #     return message

    if userText[0] == "!":
        if userText[1:] in members:
            message = f"""
                <br />
                {member_introduction[userText[1:]]}
                <br /><br />
                """                   
        elif userText[1:] in cmds["명령어"]: 
            li = cmds[userText[1:]]
            message = "<br />\n".join(li)
        else:
            message = "입력한 명령어가 존재하지 않습니다."

        return message


    text_arr = tokenizer.tokenize(userText)
    input_ids, input_mask, segment_ids = bert_to_array.transform([" ".join(text_arr)])


    # 예측
    with graph.as_default():
        with sess.as_default():
            inferred_tags, slots_score = model.predict_slots(
                [input_ids, input_mask, segment_ids], tags_to_array
            )

    # 결과 체크
    print("text_arr:", text_arr)
    print("inferred_tags:", inferred_tags[0])
    print("slots_score:", slots_score[0])

    # 슬롯에 해당하는 텍스트를 담을 변수 설정
    slot_text = {k: "" for k in app.slot_dict}

    # 슬롯태깅 실시
    for i in range(0, len(inferred_tags[0])):
        if slots_score[0][i] >= app.score_limit:
            catch_slot(i, inferred_tags, text_arr, slot_text)
        else:
            print("something went wrong!")

    # 메뉴판의 이름과 일치하는지 검증
    for k in app.slot_dict:
        for x in dic[k]:
            x = x.lower().replace(" ", "\s*")
            m = re.search(x, slot_text[k])
            if m:
                app.slot_dict[k].append(m.group())

    print(app.slot_dict)


    empty_slot = [menu[k] for k in app.slot_dict if not app.slot_dict[k]]
    log = datetime.datetime.now()

    if ('음료' in empty_slot and '푸드' in empty_slot): #음료와 푸드 이름이 인식 안된 상태
        message = '음료 혹은 푸드가 인식되지 않았습니다. 다시 입력해주세요'

    elif ('음료' not in empty_slot and '푸드' not in empty_slot): # 음료 푸드 둘 다 인식이 된 상태
        if '시럽' not in empty_slot:
            # ['음료 사이즈', '음료 수량', '음료 온도', '푸드','푸드 수량','시럽 수량']
            if ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot and '음료 온도' in empty_slot):
                # ['음료 사이즈', '음료 수량']
                # ['음료 사이즈', '음료 수량', '음료 온도']
                # ['음료 사이즈', '음료 수량', '푸드']
                # ['음료 사이즈', '음료 수량', '푸드 수량'] : 푸드 문제 발생
                # ['음료 사이즈', '음료 수량', '시럽 수량'] : 시럽 수량 문제 발생
                # ['음료 사이즈', '음료 수량', '음료 온도', '푸드','푸드 수량','시럽 수량']
                if '푸드 수량' in empty_slot:
                    message = '온도, 사이즈, 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '온도, 사이즈, 수량을 입력해주세요'

            elif ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot):
                # ['음료 사이즈', '음료 수량']
                # ['음료 사이즈', '음료 수량', '음료 온도']
                # ['음료 사이즈', '음료 수량', '푸드']
                # ['음료 사이즈', '음료 수량', '푸드 수량'] : 푸드 문제 발생
                # ['음료 사이즈', '음료 수량', '시럽 수량'] : 시럽 수량 문제 발생
                # ['음료 사이즈', '음료 수량', '음료 온도', '푸드','푸드 수량','시럽 수량']
                if '푸드 수량' in empty_slot:
                    message = '사이즈, 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '사이즈, 수량을 입력해주세요'
                
            # 사이즈만 있을 때
            elif ('음료 온도' in empty_slot and '음료 수량' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '온도, 수량을 입력해주세요'


            # 수량 만 있을 때
            elif ('음료 온도' in empty_slot and '음료 사이즈' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 사이즈, 푸드 수량을 입력해주세요'
                else:
                    message = '온도, 사이즈을 입력해주세요'       

            # 온도, 사이즈만 있을 때
            elif ('음료 수량' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '음료 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '수량을 입력해주세요'

            # 온도 수량만 있을 때
            elif ('음료 사이즈' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '사이즈, 푸드 수량을 입력해주세요'
                else:
                    message = '사이즈을 입력해주세요'

            # 사이즈, 수량만 있을 때
            elif ('음료 온도' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 푸드 수량을 입력해주세요'
                else:
                    message = '온도을 입력해주세요'
            
            elif ('시럽 수량' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '시럽 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '시럽 수량을 입력해주세요'
            else :
                if '푸드 수량' in empty_slot:
                    message = '푸드 수량, 푸드 수량을 입력해주세요'
                else:
                    if userText.strip().startswith("예"):
                        message = "주문이 완료되었습니다."
                        with open(os.path.join(data_path,f'custinfo/{log}.txt',),'w') as f:
                            f.write(order_list(app,menu))
                    elif userText.strip().startswith("아니오"):
                        message = "다시 주문 받겠습니다(초기화)"
                        # 재주문을 위해 슬롯 초기화
                        init_app(app)
                    else:
                        message = check_order_msg(app, menu)
                    

        elif ('시럽' not in empty_slot and '시럽 수량' not in empty_slot):
            if ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot and '음료 온도' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 사이즈, 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '온도, 사이즈, 수량을 입력해주세요'

            elif ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '사이즈, 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '사이즈, 수량을 입력해주세요'       

            # 사이즈만 있을 때
            elif ('음료 온도' in empty_slot and '음료 수량' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '온도, 수량을 입력해주세요'


            # 수량 만 있을 때
            elif ('음료 온도' in empty_slot and '음료 사이즈' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 사이즈, 푸드 수량을 입력해주세요'
                else:
                    message = '온도, 사이즈을 입력해주세요'


            # 온도, 사이즈만 있을 때
            elif ('음료 수량' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '수량, 푸드 수량을 입력해주세요'
                else:
                    message = '수량을 입력해주세요'

            # 온도 수량만 있을 때
            elif ('음료 사이즈' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '사이즈, 푸드 수량을 입력해주세요'
                else:
                    message = '사이즈을 입력해주세요'

            # 사이즈, 수량만 있을 때
            elif ('음료 온도' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 푸드 수량을 입력해주세요'
                else:
                    message = '온도을 입력해주세요'

            else :
                if '푸드 수량' in empty_slot:
                    message = '푸드 수량을 입력해주세요'
                else:
                    if userText.strip().startswith("예"):
                        message = "주문이 완료되었습니다."
                        with open(os.path.join(data_path,f'custinfo/{log}.txt',),'w') as f:
                            f.write(order_list(app,menu))
                    elif userText.strip().startswith("아니오"):
                        message = "다시 주문 받겠습니다(초기화)"
                        # 재주문을 위해 슬롯 초기화
                        init_app(app)
                    else:
                        message = check_order_msg(app, menu)

        else:
            if ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot and '음료 온도' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 사이즈, 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '온도, 사이즈, 수량을 입력해주세요'

            # 온도만 있을 때
            elif ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '사이즈, 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '사이즈, 수량을 입력해주세요'

            # 사이즈만 있을 때
            elif ('음료 온도' in empty_slot and '음료 수량' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 수량, 푸드 수량을 입력해주세요'
                else:
                    message = '온도, 수량을 입력해주세요'

            # 수량 만 있을 때
            elif ('음료 온도' in empty_slot and '음료 사이즈' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 사이즈, 푸드 수량을 입력해주세요'
                else:
                    message = '온도, 사이즈을 입력해주세요'

            # 온도, 사이즈만 있을 때
            elif ('음료 수량' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '수량, 푸드 수량을 입력해주세요'
                else:
                    message = '수량을 입력해주세요'

            # 온도 수량만 있을 때
            elif ('음료 사이즈' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '사이즈, 푸드 수량을 입력해주세요'
                else:
                    message = '사이즈을 입력해주세요'

            # 사이즈, 수량만 있을 때
            elif ('음료 온도' in empty_slot):
                if '푸드 수량' in empty_slot:
                    message = '온도, 푸드 수량을 입력해주세요'
                else:
                    message = '온도을 입력해주세요'

            else :
                if '푸드 수량' in empty_slot:
                    message = '푸드 수량을 입력해주세요'
                else:
                    if userText.strip().startswith("예"):
                        message = "주문이 완료되었습니다."
                        with open(os.path.join(data_path,f'custinfo/{log}.txt',),'w') as f:
                            f.write(order_list(app,menu))
                    elif userText.strip().startswith("아니오"):
                        message = "다시 주문 받겠습니다(초기화)"
                        # 재주문을 위해 슬롯 초기화
                        init_app(app)
                    else:
                        message = check_order_msg(app, menu)


    elif ('음료' not in empty_slot and '푸드' in empty_slot): #푸드 이름이 인식 안된 상태 or 음료만 주문하는 상태
        # ['음료 사이즈', '음료 수량', '음료 온도', '푸드','푸드 수량','시럽','시럽 수량']
        if '시럽' not in empty_slot:
            # ['음료 사이즈', '음료 수량', '음료 온도', '푸드','푸드 수량','시럽 수량']\
            if ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot and '음료 온도' in empty_slot  and '시럽 수량' in empty_slot):
                message = '온도, 사이즈, 수량, 시럽 수량을 입력해주세요'
                
            elif ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot and '음료 온도' in empty_slot):
                message = '온도, 사이즈, 수량을 입력해주세요'

            elif ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot):
                message = '사이즈, 수량, 시럽 수량을 입력해주세요'

            # 사이즈만 있을 때
            elif ('음료 온도' in empty_slot and '음료 수량' in empty_slot):
                message = '온도, 수량, 시럽 수량을 입력해주세요'

            # 수량 만 있을 때
            elif ('음료 온도' in empty_slot and '음료 사이즈' in empty_slot):
                message = '온도, 사이즈, 시럽 수량을 입력해주세요'

            # 온도, 사이즈만 있을 때
            elif ('음료 수량' in empty_slot):
                message = '수량, 시럽 수량을 입력해주세요'

            # 온도 수량만 있을 때
            elif ('음료 사이즈' in empty_slot):
                message = '사이즈, 시럽 수량을 입력해주세요'

            # 사이즈, 수량만 있을 때
            elif ('음료 온도' in empty_slot):
                message = '온도, 시럽 수량을 입력해주세요'
            
            elif ('시럽 수량' in empty_slot):
                message = '시럽 수량을 입력해주세요'
            else :
                if userText.strip().startswith("예"):
                    message = "주문이 완료되었습니다."
                    with open(os.path.join(data_path,f'custinfo/{log}.txt',),'w') as f:
                        f.write(order_list(app,menu))
                elif userText.strip().startswith("아니오"):
                    message = "다시 주문 받겠습니다(초기화)"
                        # 재주문을 위해 슬롯 초기화
                    init_app(app)
                else:
                    message = check_order_msg(app, menu)

        elif ('시럽' not in empty_slot and '시럽 수량' not in empty_slot):
            if ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot and '음료 온도' in empty_slot):
                message = '온도, 사이즈, 수량을 입력해주세요'

            elif ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot):
                message = '사이즈, 수량을 입력해주세요'

            # 사이즈만 있을 때
            elif ('음료 온도' in empty_slot and '음료 수량' in empty_slot):
                message = '온도, 수량을 입력해주세요'

            # 수량 만 있을 때
            elif ('음료 온도' in empty_slot and '음료 사이즈' in empty_slot):
                message = '온도, 사이즈을 입력해주세요'

            # 온도, 사이즈만 있을 때
            elif ('음료 수량' in empty_slot):
                message = '수량을 입력해주세요'

            # 온도 수량만 있을 때
            elif ('음료 사이즈' in empty_slot):
                message = '사이즈을 입력해주세요'

            # 사이즈, 수량만 있을 때
            elif ('음료 온도' in empty_slot):
                message = '온도을 입력해주세요'

            else :
                if userText.strip().startswith("예"):
                    message = "주문이 완료되었습니다."
                    with open(os.path.join(data_path,f'custinfo/{log}.txt',),'w') as f:
                        f.write(order_list(app,menu))
                elif userText.strip().startswith("아니오"):
                    message = "다시 주문 받겠습니다(초기화)"
                    # 재주문을 위해 슬롯 초기화
                    init_app(app)
                else:
                    message = check_order_msg(app, menu)

        else:
            if ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot and '음료 온도' in empty_slot):
                message = '온도, 사이즈, 수량을 입력해주세요'

            # 온도만 있을 때
            elif ('음료 사이즈' in empty_slot and '음료 수량' in empty_slot):
                message = '사이즈, 수량을 입력해주세요'

            # 사이즈만 있을 때
            elif ('음료 온도' in empty_slot and '음료 수량' in empty_slot):
                message = '온도, 수량을 입력해주세요'

            # 수량 만 있을 때
            elif ('음료 온도' in empty_slot and '음료 사이즈' in empty_slot):
                message = '온도, 사이즈을 입력해주세요'

            # 온도, 사이즈만 있을 때
            elif ('음료 수량' in empty_slot):
                message = '수량을 입력해주세요'

            # 온도 수량만 있을 때
            elif ('음료 사이즈' in empty_slot):
                message = '사이즈을 입력해주세요'

            # 사이즈, 수량만 있을 때
            elif ('음료 온도' in empty_slot):
                message = '온도을 입력해주세요'

            else :
                if userText.strip().startswith("예"):
                    message = "주문이 완료되었습니다."
                    with open(os.path.join(data_path,f'custinfo/{log}.txt',),'w') as f:
                        f.write(order_list(app,menu))
                elif userText.strip().startswith("아니오"):
                    message = "다시 주문 받겠습니다(초기화)"
                    # 재주문을 위해 슬롯 초기화
                    init_app(app)
                else:
                    message = check_order_msg(app, menu)

    elif ('푸드' not in empty_slot and '음료' in empty_slot): #음료 이름이 인식 안된 상태 or 푸드만 주문하는 상태
        #음료 이름이 인식 안된 상태 or 푸드만 주문하는 상태
        # ['음료', '음료 사이즈', '음료 수량', '음료 온도', '푸드 수량','시럽','시럽 수량']
        message = '푸드 인식'
        if '푸드 수량' in empty_slot:
            # 푸드를 주문하려고 했으나 인식이 안된 상태
            # ['음료', '음료 사이즈', '음료 수량', '음료 온도', '푸드 수량','시럽','시럽 수량']
            message = '푸드 인식 수량을 입력해주세요'
        else:
            # ['음료', '음료 사이즈', '음료 수량', '음료 온도','시럽','시럽 수량']
            if userText.strip().startswith("예"):
                message = "주문이 완료되었습니다."
                with open(os.path.join(data_path,f'custinfo/{log}.txt',),'w') as f:
                    f.write(order_list(app,menu))
            elif userText.strip().startswith("아니오"):
                message = "다시 주문 받겠습니다(초기화)"
                # 재주문을 위해 슬롯 초기화
                init_app(app)
            else:
                message = check_order_msg(app, menu)


                

    return message

def order_list(app,menu):
    order = []
    for k, v in app.slot_dict.items():
        try:
            if len(v) == 1:
                order.append(f"{menu[k]}: {v[0]}")
            else:
                order.append(f"{menu[k]}: {', '.join(v)}")
        except:
            order.append(f"{menu[k]}: {'-'}")

    message = f"{set(order)}"

    return message

def check_order_msg(app, menu):
    order = []
    for k, v in app.slot_dict.items():
        try:
            if len(v) == 1:
                order.append(f"{menu[k]}: {v[0]}")
            else:
                order.append(f"{menu[k]}: {', '.join(v)}")
        except:
            order.append(f"{menu[k]}: {None}")
 

    order = "<br />\n".join(set(order))

    message = f"""
        주문 확인하겠습니다.<br />
        ===================<br />
        {order}
        <br />===================<br />
        이대로 주문 완료하시겠습니까? (예 or 아니오)
        """

    return message

def init_app(app):
    app.slot_dict = {
        'beverage': [],
        'temperature':[],
        'size':[],
        'quantity':[],
        'syrup':[],
        'syrup_quantity':[],
        'food':[],
        'food_quantity':[]
        }


    app.ask_syrup = False


def catch_slot(i, inferred_tags, text_arr, slot_text):
    if not inferred_tags[0][i] == "O":
        word_piece = re.sub("_", " ", text_arr[i])
        if word_piece == 'ᆫ':
            word = slot_text[inferred_tags[0][i]]
            slot_text[inferred_tags[0][i]] = word[:-1]+chr(ord(word[-1])+4)
        else:
            if word_piece == "오" and inferred_tags[0][i] == "vegetable":
                slot_text[inferred_tags[0][i]] += "오이"
            else:
                slot_text[inferred_tags[0][i]] += word_piece
