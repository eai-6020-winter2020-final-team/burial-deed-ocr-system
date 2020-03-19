from flask import Flask
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

from app import config

import logging

# from MODULE import main_function

app = Flask(
	"CemeteryTest",
	template_folder='./templates/',
	static_folder='./',
	static_url_path='',
)
app.config.from_object(config)

handler = logging.FileHandler('changes.log')
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)


db = SQLAlchemy(app)
migrate = Migrate(app, db)
loginManager = LoginManager(app)
loginManager.login_view = 'login'

from app import routes, tables

db.create_all()

if __name__ == '__main__':
	app.run()
