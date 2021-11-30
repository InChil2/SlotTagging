![header](https://capsule-render.vercel.app/api?type=waving&color=random&text=Slot-Tagging&animation=fadeIn&fontColor=B5B5B6)

<h5 align='center'> Using Tech </h5>

<p align='center'>
  <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=Jupyter&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=Flask&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/Selenium-43B02A?style=flat-square&logo=Selenium&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=white"/></a>&nbsp
</p>



#### 팀장, 총괄
##### [이슬](https://github.com/seuly1203)
![](https://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=seuly1203&theme=monokai)
#### 챗봇 알고리즘, BERT 모델링
##### [박민아](https://github.com/parkmina365)
![](https://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=parkmina365&theme=monokai)
#### 웹코딩, PPT
##### [정하림](https://github.com/hharimjung)
![](https://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=hharimjung&theme=monokai)
#### 웹코딩, 크롤링
##### [오수문](https://github.com/sumunoh)
![](https://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=sumunoh&theme=monokai)
#### 기술 총괄
##### [신인철](https://github.com/InChil2)
![](https://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=InChil2&theme=monokai)

#### 2021.10.18(월) Start
----------------------------------------------------------------------------------------------------------------------------------------------------
## 별다방 주문 챗봇
![image](https://user-images.githubusercontent.com/86215518/143837501-67352039-9096-482b-92de-962c4f45fe11.jpg)


### 목차
![image](https://user-images.githubusercontent.com/86215518/143837540-25d92f22-4c62-4823-b215-5d552f44a506.jpg)



### 01. 서비스 기획 목적
#### 기존 사이렌오더 서비스가 존재함에도 불구하고 챗봇 서비스를 개발하게 된 이유 = 사이렌오더 앱 설치 및 이용에 대한 불편함, 어려움
![image](https://user-images.githubusercontent.com/86215518/143963783-573a9b96-a283-4fd1-bd5f-4cbe89b0df6d.jpg)




### 02. 서비스 시나리오
: 핵심어 추출로 사용자가 원하는 주문을 챗봇이 정확하게 파악할 수 있도록 함.
: ex) 아메리카노, 라떼, .. = 음료 / 쿠키, 케이크, .. = 음식 / 따뜻하게, 차갑게, .. = 온도 / ..
![image](https://user-images.githubusercontent.com/86215518/143963963-c0a4ab21-d0bd-4377-b0de-9c6be1d91c7c.jpg)

#### 서비스 시나리오(기본)
![image](https://user-images.githubusercontent.com/86215518/143965388-fcab1642-b576-418c-98ce-10f45d21f671.jpg)

#### 서비스 시나리오(예외 발생시)
: 빈 슬롯 발생, 예외 발생 시 필요한 시나리오 구상(챗봇이 인식할 수 없는 주문이 들어올 경우, 어떻게 처리할 것인가에 대한 방안 마련)
![image](https://user-images.githubusercontent.com/86215518/143964415-28852eb3-e3d0-4c14-8598-69764f438fbd.jpg)




### 03. 파인튜닝 방법
#### 핵심기술사용 : BERT
: pre-training 된 BERT 에게 fine-tuning 을 추가로 시켜 해당 프로젝트의 목적에 맞는 모델 생성
![image](https://user-images.githubusercontent.com/86215518/143964504-6ca65eae-3fdb-4f44-b456-541d1218b6fb.jpg)

#### 파인튜닝방법 : 슬롯태깅
![image](https://user-images.githubusercontent.com/86215518/143964574-87b4def6-6981-42c3-80be-5665d5244ea5.jpg)



### 04. 데이터 수집 방안
![image](https://user-images.githubusercontent.com/86215518/143964886-bdb217db-0460-4bb2-9090-a281e14bbea5.jpg)



### 05. 서비스 제작 계획
#### 제작 순서
: flask, figma, HTML, CSS 를 사용하여 웹서비스로 챗봇 시스템 구현
![image](https://user-images.githubusercontent.com/86215518/143964983-446a9da5-1162-4fe7-9a66-016a05ab905e.jpg)

#### 서비스 제작 계획 : 챗봇 구현(대시보드)
![image](https://user-images.githubusercontent.com/86215518/143965023-e2920c86-c6ed-4f81-9c42-1de3ce4f36e7.jpg)

#### 서비스 제작 계획 : 챗봇 시연(이미지)
![image](https://user-images.githubusercontent.com/86215518/143965045-afac7883-0313-4d32-8b51-7ad082c075f5.jpg)




### 06. 서비스 기대효과
1) 소비자 맞춤 서비스 제공
2) 신규 고객 유입 : 스타벅스 어플을 설치 및 가입하지않아도 쉽게 이용 가능하므로 신규 고객의 유입 가능성을 기대해볼 수 있음.
3) 접근성 향상 : 별도의 앱 설치가 필요없으므로 접근성이 용이함.


![image](https://user-images.githubusercontent.com/86215518/143965240-2b3a95c3-22c0-45e9-b72e-610a41c5b7ba.jpg)

----------------------------------------------------------------------------------------------------------------------------------------------------

# 버트 미세조정 - 슬롯 태깅  
  
1. pretrained BERT 모델을 모듈로 export  

    - ETRI에서 사전훈련한 BERT의 체크포인트를 가지고 BERT 모듈을 만드는 과정.  
    - `python export_korbert/bert_to_module.py -i {체크포인트 디렉토리} -o {output 디렉토리}`   
    - 예시: `python export_korbert/bert_to_module.py -i /content/drive/MyDrive/004_bert_eojeol_tensorflow -o /content/drive/MyDrive/bert-module`  
  
2. 데이터 준비

    1) 모델을 훈련하기 위해 필요한 seq.in, seq.out이라는 2가지 파일을 만드는 과정  
       - `python prepare_data.py process_file({변환할 data.txt 파일 경로}, {아웃풋 저장 경로}), process_line(문장, 토큰)`
       - `'홍길동'을 입력하면 '홍', '길', '동' 으로 분류가 되어 출력이 되므로 좀 더 정확한 분류가 이루어짐`
      
    2) 제작한 seq.in, seq.out을 train, validation, test set이 8:1:1 비율이 되도록 나누기
       - `split_new.py`

3. Fine-tuing 훈련  

    - `python train.py -t {train set 디렉토리} -v {validation set 디렉토리} -s {model이 저장될 디렉토리} -e {epoch의 수} -bs {batch size의 수}`
  
4. 모델 평가  

    - `python eval.py -m {훈련된 model이 저장된 디렉토리} -d {test set 디렉토리}`  
    - 테스트의 결과는 -m에 넣어준 model 디렉토리 아래의 `test_results`에 저장됨 
  
5. Inference (임의의 문장을 모델에 넣어보기)  

    - `python inference.py -m {훈련된 model이 저장된 디렉토리}`  
    - 예시: `python inference.py -m saved_model/`   
    - "Enter your sentence:"라는 문구가 나오면 모델에 넣어보고 싶은 문장을 넣어 주면 됨  
    - 'quit', '종료', '그만', '멈춰', 'stop'라는 입력을 넣어 주면 종료
  
