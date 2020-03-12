from flask import Flask
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

from app import config

# from MODULE import main_function

app = Flask("CemeteryTest")
app.config.from_object(config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
loginManager = LoginManager(app)
loginManager.login_view = 'login'

from app import routes, tables

db.create_all()

if __name__ == '__main__':
	app.run()
