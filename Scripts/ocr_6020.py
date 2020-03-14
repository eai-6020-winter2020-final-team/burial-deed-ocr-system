import pytesseract
import cv2
import numpy as np
import os
import re
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf


class CardOcr(object):
    """
    OCR for card recognition pipeline: card type classification, form classification, crop and read.
    """
    def __init__(self, img_path):
        self.image = cv2.imread(img_path, 1)

    def show_image(self):
        """

        :return: image
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(self.image)
        fig.show()

    def card_type(self):
        """
        find card type
        :return: string
        """
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        im = image[180:280, 0:200]
        text = pytesseract.image_to_string(im)
        if 'No' in text:
            return 'Deed'
        else:
            im = image[150:300, 0:500]
            text = pytesseract.image_to_string(im)
            if 'No' in text:
                return 'Deed'
            else:
                return 'Burial'

    def img_to_arr(self, x, y):
        """
        Convert image to array
        :param x: shape[0]
        :param y: shape[1]
        :return:
        """
        img = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)).convert('L')
        if img.size[0] != x or img.size[1] != y:
            img = img.resize((x, y))

        arr = []

        for i in range(y):
            for j in range(x):
                pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
                arr.append(pixel)
        return arr

    def form_type(self):
        """
        Form classification for burial cards
        :return: string
        """
        model_form = tf.keras.models.load_model('model_form.h5')
        input_img = np.array(self.img_to_arr(500, 500))
        input_img = input_img.reshape(1, 500, 500, 1)
        ret = model_form.predict(input_img)
        rel = np.where(ret[0] == np.max(ret[0]))
        if rel[0] == [0]:
            return 'A'
        elif rel[0] == [1]:
            return 'B'
        elif rel[0] == [2]:
            return 'C'
        else:
            return 'NaN'

    def flag_hw(self):
        """
        flag handwriting
        :return: string
        """
        model_hw = tf.keras.models.load_model('model_hw.h5')
        input_img = np.array(self.img_to_arr(500, 500))
        input_img = input_img.reshape(1, 500, 500, 1)
        ret = model_hw.predict(input_img)
        rel = np.where(ret[0] == np.max(ret[0]))
        if rel[0] == [0]:
            return 'N'
        else:
            return 'Y'

    def flag_f(self):
        """
        flag fraction
        :return: string
        """
        model_f = tf.keras.models.load_model('model_f.h5')
        input_img = np.array(self.img_to_arr(500, 500))
        input_img = input_img.reshape(1, 500, 500, 1)
        ret = model_f.predict(input_img)
        rel = np.where(ret[0] == np.max(ret[0]))
        if rel[0] == [0]:
            return 'N'
        else:
            return 'Y'


def main():
    """
    create dataframe
    :return: csv file
    """
    path = 'All_Data/'
    files = sorted(os.listdir(path))
    ret = []

    for i in files:
        img_info = [i]
        new_card = CardOcr(path + i)

        hw = new_card.flag_hw()
        img_info.append(hw)

        fraction = new_card.flag_f()
        img_info.append(fraction)

        cla = new_card.card_type()
        img_info.append(cla)
        if cla == 'Burial':
            form = new_card.form_type()
            img_info.append(form)
        else:
            img_info.append('NaN')

        ret.append(img_info)

    df = pd.DataFrame(ret, columns=['file_name', 'handwriting', 'fraction', 'card_type', 'form_type'])

    return df.to_csv('classification.csv')


if __name__ == '__main__':
    main()






