import numpy as np
from matplotlib import pyplot as plt

def main_function(file) -> dict:
	"""
	Main Recognition function
	Inputs:
		file:File (Image File Object)
	Outputs:
		Dict (Recognition Result)
	"""
	raw_image_arr = plt.imread(file)
	image_arr = preprocess(raw_image_arr)
	recognizer_func = classify_function(image_arr)
	result_dict = recognizer_func(image_arr)
	return result_dict


def preprocess(img) -> np.ndarray:
	"""
	Preprocess Image
	Inputs:
		img:ndarray (Image numpy.ndarray Object)
	Outputs:
		ndarray (Cleaned Image)
	"""
	pass


def classify_function(img):
	"""
	Classify Images into 3 groups
	Inputs:
		img:ndarray (Image numpy.ndarray Object)
	Outputs:
		function (Function used for recognition)
	"""
	pass


def recognition_a(img):
	pass


def recognition_b(img):
	pass


def recognition_c(img):
	pass