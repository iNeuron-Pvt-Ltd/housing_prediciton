from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    try:
        return "Hello World"
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run()