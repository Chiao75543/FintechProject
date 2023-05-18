from flask import Flask , request, jsonify,render_template
from flask_restful import Api
import pymysql
import traceback
import jwt
import time
from server import app

# app = Flask(__name__)
api = Api(app)


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
 