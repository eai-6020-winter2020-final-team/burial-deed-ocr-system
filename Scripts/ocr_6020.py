import pytesseract
import cv2
import numpy as np
import os
import re
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf


def image_to_conf(image, text_type, n=0):
    """
    Using tesseract to get text and confidence
    """
    if text_type == 0:
        # to recognize a line of text
        data = pytesseract.image_to_data(image, config='-l eng --psm 7', output_type='data.frame')
    elif text_type == 1:
        # to recognize a block of text
        data = pytesseract.image_to_data(image, config='-l eng --oem 1 --psm 6', output_type='data.frame')
    else:
        data = pytesseract.image_to_data(image, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789',
                                         output_type='data.frame')
    text_list = []
    sum_conf = 0
    len_conf = 0
    for i in range(len(data)):
        if data.iloc[i]['conf'] > n:
            text_list.append(str(data.iloc[i]['text']))
            sum_conf += len(str(data.iloc[i]['text'])) * data.iloc[i]['conf']
            len_conf += len(str(data.iloc[i]['text']))
    if len_conf == 0:
        conf = 0
    else:
        conf = sum_conf / len_conf
    return ' '.join(text_list), conf


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


class CardDetectA(CardOcr):
    """
    OCR for card recognition
    """
    def __init__(self, img_path):
        super().__init__(img_path)
        self.image = cv2.imread(img_path, 1)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # detect first line and draw Auxiliary line
    def first_line(self):
        # Image Binarization
        ret, thresh1 = cv2.threshold(self.gray, 200, 255, cv2.THRESH_BINARY)
        # Median filter
        blur = cv2.medianBlur(thresh1, 3)  # ksize: 3*3
        blur = cv2.medianBlur(blur, 3)
        h, w = self.gray.shape
        # horizontal line
        horizontal_lines = []
        for i in range(h - 1):
            # find the split line
            if abs(np.mean(blur[i, :]) - np.mean(blur[i + 1, :])) > 100:
                # plot the line
                horizontal_lines.append([0, i, w, i])
        if horizontal_lines and horizontal_lines[0][1] < 150:
            first_line = horizontal_lines[0]
        else:
            first_line = [0, 104, 930, 104]
        # plot lines
        lines = [[23, 30, 23, 239], [930, 30, 930, 239]]
        first_line[1] -= 55
        first_line[3] -= 55
        lines.append(first_line)
        for l in lines:
            self.image = cv2.line(self.image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2)

        return self.image

    # detect cell
    def find_form(self):
        image = self.first_line()
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Image Binarization
        binary = cv2.adaptiveThreshold(~self.gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
        rows, cols = binary.shape
        scale = 20

        # detect horizontal line
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (cols // scale, 1))
        eroded = cv2.erode(binary, kernel, iterations=2)
        dilatedcol = cv2.dilate(eroded, kernel, iterations=2)

        # detect vertical line
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, rows // scale))
        eroded = cv2.erode(binary, kernel, iterations=2)
        dilatedrow = cv2.dilate(eroded, kernel, iterations=2)

        # get vertex
        vertex = cv2.bitwise_and(dilatedcol, dilatedrow)

        # plot cell
        merge = cv2.add(dilatedcol, dilatedrow)
        return vertex, merge

    # get the coordinate
    def get_coordinate(self):
        a, b = self.find_form()
        ys, xs = np.where(a > 0)
        # create coordinate
        x_list, y_list = [], []

        # sort list
        xs, ys = np.sort(xs), np.sort(ys)

        for i in range(len(xs) - 1):
            if xs[i+1] - xs[i] > 20:
                x_list.append(xs[i])
        x_list.append(xs[i])
        if x_list[0] > 30:
            x_list.append(24)
        x_list = sorted(x_list)

        for i in range(len(ys) - 1):
            if ys[i+1] - ys[i] > 20:
                y_list.append(ys[i])
        y_list.append(ys[i])
        try:
            y_list[1] += 18
        except IndexError:
            pass

        try:
            y_list[2] -= 5

        except IndexError:
            pass

        return x_list, y_list

    # find cell
    def cell_detect(self):

        x_list, y_list = self.get_coordinate()
        # print(x_list)
        # crop the image by second line
        try:
            self.image = self.image[0:y_list[2], 0:950].copy()
        except IndexError:
            pass
        rects = []

        try:
            first_line = [x_list[0], x_list[-3]-25, x_list[-1]]
            for i in range(0, len(first_line) - 1):
                for j in range(len(y_list[:2]) - 1):
                    rects.append((first_line[i], y_list[j], first_line[i + 1], y_list[j + 1]-20))
        except IndexError:
            pass
        if len(y_list) > 3:
            y_list.pop()
        else:
            pass

        if len(x_list) >= 8:
            x_list.pop(-4)
        else:
            pass

        for i in range(0, len(x_list) - 1):
            for j in range(1, len(y_list) - 1):
                rects.append((x_list[i], y_list[j], x_list[i + 1], y_list[j + 1]))

        # plot rect:
        for rect in rects:
            self.image = cv2.rectangle(self.image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)

        # print(rects)
        return rects

    # extract text:
    def ocr_text(self):
        rects = self.cell_detect()
        thresh = self.gray

        target = [0, 2, 3, 4, 5]
        file_name = ['Name', 'Date of interment', 'Section', 'Lot', 'GR']
        special_char = '‘’,|-_<"=;«“&—]uv'
        file_type = [0, 0, 1, 1, 2]
        threshold = [-1, -1, -1, -1, -1]
        # rect1 = rects[target[0]]
        # detect_img = thresh[rect1[1]:rect1[3], rect1[0]:rect1[2]]
        # data = pytesseract.image_to_data(detect_img, config='--psm 7', lang='eng', output_type='data.frame')
        # print(data)
        result = {}
        conf = 0
        try:
            for i in range(5):
                rect1 = rects[target[i]]
                detect_img = thresh[rect1[1]:rect1[3], rect1[0]:rect1[2]]
                name = file_name[i]
                type1 = file_type[i]
                thresh1 = threshold[i]
                text1, conf1 = image_to_conf(detect_img, type1, thresh1)
                text1 = ''.join([char for char in text1 if char not in special_char])
                conf += conf1
                if i == 0:
                    text1 = text1.lstrip('7')
                    text1 = re.sub("\s+", " ", ''.join(re.findall(r'[A-Za-z0-9]\s*',
                                                                  re.sub("[,|()|-]", " ", text1)))).upper()
                elif i == 1:
                    text1 = re.sub("\s+", " ", ''.join(re.findall(r'[A-Za-z0-9]\s*\/*', text1)))
                else:
                    text1 = re.sub('\.0',"",text1)
                    text1 = re.sub("\s+", " ", ''.join(re.findall(r'[A-Za-z0-9]\s*', text1)))

                result[name] = text1
                # print(name, ':', text1, end='\n')
            result['Avg_conf'] = round((conf / 5), 2)
        except IndexError:
            pass

        return result


def main():
    """
    create dataframe
    :return: csv file
    """
    path = 'All_Data/'
    files = sorted(os.listdir(path))
    ret = []
    temp = []

    for i in files:
        img_info = [i]
        new_card = CardDetectA(path + i)

        hw = new_card.flag_hw()
        img_info.append(hw)

        fraction = new_card.flag_f()
        img_info.append(fraction)

        cla = new_card.card_type()
        img_info.append(cla)
        if cla == 'Burial':
            form = new_card.form_type()
            img_info.append(form)
            if form == 'A':
                ret = new_card.ocr_text()
                temp.append(ret)
            else:
                pass
        else:
            img_info.append('NaN')

        ret.append(img_info)

    df = pd.DataFrame(ret, columns=['file_name', 'handwriting', 'fraction', 'card_type', 'form_type'])
    form_a = pd.DataFrame(temp, columns=['Name', 'Date of interment', 'Section', 'Lot', 'GR', 'Avg_conf'])

    return df.to_csv('classification.csv'), form_a.to_csv('result.csv')


if __name__ == '__main__':
    main()
