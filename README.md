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
    - 예시: `python inference.py --model saved_model/`   
    - "Enter your sentence:"라는 문구가 나오면 모델에 넣어보고 싶은 문장을 넣어 주면 됨  
    - 'quit', '종료', '그만', '멈춰', 'stop'라는 입력을 넣어 주면 종료
  
