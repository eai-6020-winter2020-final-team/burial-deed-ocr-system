from flask import Flask
from flask import request
from flask import render_template

import json

#from MODULE import main_function

app = Flask("CemeteryTest")

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/upload/', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		f = request.files['image_file']
	
	#return json.dumps(main_function(f))
	dic = {
		"fileName": f.filename,
		"personName": "person_name",
		"outputV1": "ov1",
		"outputV2": "ov2",
		"outputV3": "ov3",
		"outputV4": "ov4"
	}
	return json.dumps(dic)

if __name__ == '__main__':
	app.run()
