from flask_sqlalchemy import SQLAlchemy
from flask import Flask

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost:3306/api'
# db = SQLAlchemy(app)