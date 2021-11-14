# 챗봇 웹 데모  
  
https://github.com/Arraxx/new-chatbot 참고  

https://github.com/k151202/BERT_sandwich_order 참고


1. run.py
- main.py 파일을 실행시켜서 웹에서 작동시키는 파이썬 파일

2. main.py
- 챗봇의 응답을 구현하기 위한 파이썬 파일
- BERT 모델만으로는 구현이 어려운 챗봇의 예외처리 대응을 조건절로 만들어 추가
- main_v1.py: 주문 완료 요약 메세지 출력, 팀원 명령어 추가 (+ 전화번호를 입력해 고객 정보를 저장하는 기능 추가중)
- main_v2.py: 조건절 간략화, 시럽 추가여부 질문
- run.py에서 정상적으로 작동시키기 위해서는 파일명을 main.py로 저장한 후 실행해야 함
