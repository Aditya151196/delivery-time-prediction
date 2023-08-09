from flask import Flask
from src.logger import logging
from src.exception import CustomException
import os,sys

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    try:
        raise Exception("We are testing exception file")
    except Exception as e:
        exception_custom=CustomException(e,sys)
        logging.info(exception_custom.error_message)
    #logging.info("We are testing logging file")

    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)