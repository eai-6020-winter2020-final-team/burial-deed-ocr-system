from app import app, db
from app.forms import LoginForm, RegistrationForm
from app.tables import User, Record, Burial, Deed

from flask import abort, flash, make_response, Response, redirect, render_template, request, url_for
from flask_login import login_user, logout_user, current_user, login_required

import os
import json
import hashlib
from io import BytesIO
from matplotlib import pyplot as plt

from Scripts.ocr_6020 import image_ocr


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
			flash('Welcome to Cemetery Digital Management System')
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


@app.route('/records/<record_type>/')
@login_required
def record(record_type):
	if record_type in ['burials', 'deeds']:
		return render_template('records.html', navi=record_type)
	else:
		return abort(404)


@app.route('/getrecords/')
@login_required
def getrecords():
	record_type = request.args.get('RecordType')
	if record_type == 'burials':
		record_table = Burial
	elif record_type == 'deeds':
		record_table = Deed
	else:
		return abort(404)
	record_list = []
	for deed_record in record_table.query.filter().all():
		temp = {'filename': Record.query.filter(Record.id == deed_record.id).first().filename}
		temp.update(deed_record.get_dic())
		record_list.append(temp)
	return json.dumps(record_list)


@app.route('/editrecord/', methods=['POST'])
@login_required
def editrecord():
	record_dict = request.form
	record_type = Record.query.filter(Record.id == record_dict['id']).first().doctype
	if record_type == 'burial':
		record_edited = Burial.query.filter(Burial.id == record_dict['id']).first()
	else:
		record_edited = Deed.query.filter(Deed.id == record_dict['id']).first()
	record_edited.update(record_dict)
	db.session.commit()

	app.logger.info(f'User [{current_user.username} : {current_user.id}] edited record [{record_dict["id"]}]')
	return make_response('')


@app.route('/deleterecord/', methods=['POST'])
@login_required
def deleterecord():
	record_dict = request.form
	outer_record_deleted = Record.query.filter(Record.id == record_dict['id']).first()
	file_deleted = outer_record_deleted.id
	record_type = outer_record_deleted.doctype
	if record_type == 'burial':
		inner_record_deleted = Burial.query.filter(Burial.id == record_dict['id']).first()
	else:
		inner_record_deleted = Deed.query.filter(Deed.id == record_dict['id']).first()

	db.session.delete(inner_record_deleted)
	db.session.delete(outer_record_deleted)
	db.session.commit()
	os.remove(f'./uploads/{file_deleted}')

	app.logger.info(f'User [{current_user.username} : {current_user.id}] deleted record [{file_deleted}]')
	return make_response('')


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

		# Classification and OCR should return a dic
		"""
		record_dic = {}
		record_dic['id'] = file_hash
		record_dic['filename'] = image_file.filename
		record_dic['doctype'] = "burial"
		record_dic['name'] = "person_name"
		record_dic['date'] = "1900.1.1"
		record_dic['section'] = "value_section"
		record_dic['lot'] = "value_lot"
		record_dic['gr'] = "value_gr"
		"""
		img_stream = BytesIO(file_content)
		img_type = image_file.filename.split('.')[-1]
		img = plt.imread(img_stream, format=img_type)
		record_dic = image_ocr(img)
		print(record_dic)
		record_dic['id'] = file_hash
		record_dic['filename'] = image_file.filename
		record_dic['doctype'] = record_dic.pop('card_type')
		record_dic['name'] = str(record_dic.pop('Name'))
		if record_dic['form'] == 'A':
			record_dic['date'] = str(record_dic.pop('Date of interment'))
			record_dic['section'] = str(record_dic.pop('Section'))
			record_dic['lot'] = str(record_dic.pop('Lot'))
			record_dic['gr'] = str(record_dic.pop('GR'))
		elif record_dic['doctype'] == 'burial' and (record_dic['form'] == 'B' or record_dic['form'] == 'C'):
			record_dic['date'] = str(record_dic.pop('Date of Burial'))
			record_dic['section'] = str(record_dic.pop('Lot-Sec-Gr-Ter'))
			record_dic['lot'] = ''
			record_dic['gr'] = ''
		else:
			record_dic['date'] = str(record_dic.pop('Deed No. & Date'))
			record_dic['section'] = str(record_dic.pop('Lot-Sec-Gr'))
			record_dic['lot'] = ''
			record_dic['deedno'] = ''

		record = Record(record_dic)
		if record.doctype == 'burial':
			sub_record = Burial(record_dic)
		else:
			sub_record = Deed(record_dic)

		db.session.add(record)
		db.session.add(sub_record)
		db.session.commit()
		with open('./uploads/' + file_hash, 'wb') as saved_file:
			saved_file.write(file_content)

		app.logger.info(f'User [{current_user.username} : {current_user.id}] added record [{file_hash}]')
		return record_dic

	result_dic = scan_image(f)
	# return abort(500)
	return json.dumps(result_dic)


@app.route('/log/')
@login_required
def log():
	with open('changes.log') as log_file:
		log_text = log_file.read()
	return render_template('log.html', navi='log', log_text=log_text)