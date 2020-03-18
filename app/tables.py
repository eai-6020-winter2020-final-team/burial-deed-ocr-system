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
	doctype = db.Column(db.Enum('burial', 'deed'))

	def __repr__(self):
		return '<Record {}>'.format(self.id)

	def __init__(self, dic):
		self.id = dic['id']
		self.filename = dic['filename']
		self.doctype = dic['doctype']


class Burial(db.Model):
	id = db.Column(db.String(32), db.ForeignKey('record.id'), primary_key=True)
	name = db.Column(db.String(128))
	# date = db.Column(db.Date)
	date = db.Column(db.String(32))
	section = db.Column(db.String(128))
	lot = db.Column(db.String(128))
	gr = db.Column(db.String(128))

	def __repr__(self):
		return '<Burial {}>'.format(self.id)

	def __init__(self, dic):
		self.id = dic['id']
		self.update(dic)

	def update(self, dic):
		self.name = dic['name']
		self.date = dic['date']
		self.section = dic['section']
		self.lot = dic['lot']
		self.gr = dic['gr']

	def get_dic(self) -> dict:
		return {'id': self.id, 'name': self.name, 'date': self.date, 'section': self.section, 'lot': self.lot, 'gr': self.gr}


class Deed(db.Model):
	id = db.Column(db.String(32), db.ForeignKey('record.id'), primary_key=True)
	name = db.Column(db.String(128))
	# date = db.Column(db.Date)
	date = db.Column(db.String(32))
	section = db.Column(db.String(128))
	lot = db.Column(db.String(128))
	deedno = db.Column(db.String(128))

	def __repr__(self):
		return '<Deed {}>'.format(self.id)

	def __init__(self, dic):
		self.id = dic['id']
		self.update(dic)

	def update(self, dic):
		self.name = dic['name']
		self.date = dic['date']
		self.section = dic['section']
		self.lot = dic['lot']
		self.deedno = dic['deedno']

	def get_dic(self) -> dict:
		return {'id': self.id, 'name': self.name, 'date': self.date, 'section': self.section, 'lot': self.lot, 'deedno': self.deedno}
