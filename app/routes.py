from app import app, db
from app.forms import LoginForm, RegistrationForm
from app.tables import User, Record

from flask import abort, flash, Response, redirect, render_template, request, url_for
from flask_login import login_user, logout_user, current_user, login_required

import json
import hashlib
from datetime import datetime

RECORD_COLUMNS = ["id", "filename", "doctype", "name", "date", "val1", "val2", "val3"]


@app.route('/login/', methods=['GET', 'POST'])
def login():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = LoginForm()
	if form.validate_on_submit():
		user = User.query.filter_by(username=form.username.data).first()
		if user is None:
			flash('User not exist')
			return redirect(url_for('login'))
		if not user.check_password(form.password.data):
			flash('Invalid password')
			return redirect(url_for('login'))
		login_user(user)
		next_page = request.args.get('next')
		if not next_page:
			next_page = url_for('home')
		return redirect(next_page)
	return render_template('login.html', title='Sign In', form=form)


@app.route('/logout/')
def logout():
	logout_user()
	return redirect(url_for('login'))


@app.route('/register/', methods=['GET', 'POST'])
def register():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RegistrationForm()
	if form.validate_on_submit():
		user = User().set_username(form.username.data).set_password(form.password.data)
		db.session.add(user)
		db.session.commit()
		flash('Registered!')
		return redirect(url_for('login'))
	return render_template('register.html', title='Register', form=form)


@app.route('/')
@login_required
def home():
	return render_template('home.html', navi="home")


@app.route('/records/')
@login_required
def records():
	record_list = []
	for record in Record.query.filter().all():
		temp = {}
		for key in RECORD_COLUMNS:
			temp[key] = str(record.__getattribute__(key))
		record_list.append(temp)
	return render_template('records.html', navi="records", record_list=record_list)


@app.route('/upload/', methods=['POST'])
@login_required
def upload():
	f = request.files['image_file']

	def scan_image(image_file) -> dict:
		file_content = image_file.stream.read()
		file_hash = hashlib.md5(file_content).hexdigest()

		record = Record.query.filter_by(id=file_hash).first()
		if record is not None:
			return {"Error": "This file already exists on server"}

		filename = image_file.filename
		doctype = "deed"
		name = "person_name"
		date = datetime.date(datetime(2000, 1, 1))
		val1 = "value1"
		val2 = "value2"
		val3 = "value3"

		record = Record()
		record.id = file_hash
		record.filename = filename
		record.doctype = doctype
		record.name = name
		record.date = date
		record.val1 = val1
		record.val2 = val2
		record.val3 = val3
		db.session.add(record)
		db.session.commit()
		with open('./uploads/'+file_hash, 'wb') as saved_file:
			saved_file.write(file_content)

		return {
			"filename": filename,
			"doctype": doctype,
			"name": name,
			"date": str(date),
			"val1": val1,
			"val2": val2,
			"val3": val3
		}
	#return abort(500)
	return json.dumps(scan_image(f))
