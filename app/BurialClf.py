import numpy as np
import joblib
from skimage import color
from skimage.transform import resize
from skimage.feature import hog

clf = joblib.load('./app/BurialClf.pkl')


def burial_clf(img: np.ndarray) -> str:
	if len(img.shape) > 2:
		img = color.rgb2gray(img)
	resizedImg = resize(img, (1000, 600))
	hogDescriptor = hog(resizedImg, orientations=12, pixels_per_cell=(24, 24), cells_per_block=(1, 1))
	clfResult = clf.predict(hogDescriptor.reshape(1, -1))
	return clfResult[0]
