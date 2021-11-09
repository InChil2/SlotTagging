from app.main import app

if __name__ == "__main__":
    # flask-ngrok 으로 실행 시
    app.run()

    # 로컬에서 실행시
    # app.run(port=6006, debug=True)