from app import db, loginManager
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin


class User(UserMixin, db.Model):
	id = db.Column(db.Integer, primary_key=True)
	username = db.Column(db.String(64), index=True, unique=True)
	password_hash = db.Column(db.String(128))

	def __repr__(self):
		return '<User {}>'.format(self.username)

	def set_username(self, username: str):
		self.username = username
		return self

	def set_password(self, password):
		self.password_hash = generate_password_hash(password)
		return self

	def check_password(self, password):
		return check_password_hash(self.password_hash, password)


@loginManager.user_loader
def load_user(user_id):
	return User.query.get(int(user_id))


class Record(db.Model):
	id = db.Column(db.String(32), primary_key=True)
	filename = db.Column(db.String(128))
	doctype = db.Column(db.Enum('deed', 'burial'))
	name = db.Column(db.String(128))
	date = db.Column(db.Date)
	val1 = db.Column(db.String(128))
	val2 = db.Column(db.String(128))
	val3 = db.Column(db.String(128))

	def __repr__(self):
		return '<Record {}>'.format(self.id)
