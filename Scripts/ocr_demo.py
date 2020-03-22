import pytesseract
import cv2
import numpy as np
import os
import re
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K


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


def find_form(image):
    """
    Find Form of image
    update: 2020.3.20
    """
    # Binarization
    binary = cv2.adaptiveThreshold(~image, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    rows, cols = binary.shape
    scale = 20
    # horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=2)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=2)

    # vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=2)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=2)

    # get vertex
    vertex = cv2.bitwise_and(dilatedcol, dilatedrow)

    # get cell
    merge = cv2.add(dilatedcol, dilatedrow)
    return vertex, merge


def img_to_arr(img, x, y):
    """
    Convert image into array
    update: 2020.3.20
    """
    """Eric adaption"""
    if len(img.shape) > 2:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('L')
    else:
        img = Image.fromarray(img).convert('L')
    if img.size[0] != x or img.size[1] != y:
        img = img.resize((x, y))

    arr = []

    for i in range(y):
        for j in range(x):
            pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
            arr.append(pixel)

    return arr


def keras_input_data(data, img_width, img_height):
    """
    Convert input image into keras input data
    update: 2020.3.20
    """
    # Load dataset
    input_data = np.array(data)

    # Reshape data based on channels first / channels last strategy.
    # This is dependent on whether you use TF, Theano or CNTK as backend.
    # Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    if K.image_data_format() == 'channels_first':
        input_data = input_data.reshape(input_data.shape[0], 1, img_width, img_height)
    else:
        input_data = input_data.reshape(input_data.shape[0], img_width, img_height, 1)

    # Parse numbers as floats
    input_data = input_data.astype('float32')

    # Normalize data
    input_data = input_data / 255

    return input_data


class CardOcr(object):
    """
    OCR for card recognition pipeline: card type classification, form classification, crop and read.
    """

    # def __init__(self, img_path):
    #    self.image = cv2.imread(img_path, 1)
    #    self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    """Adaptions made by Eric"""

    def __init__(self, img):
        if len(img.shape) > 2:
            self.image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # denoising:
            denoise = cv2.fastNlMeansDenoisingColored(self.image, None, 3, 3, 7, 21)
            self.gray = cv2.cvtColor(denoise, cv2.COLOR_RGB2GRAY)
        else:
            self.image = img
            self.gray = img

    def show_image(self, code=cv2.COLOR_BGR2RGB):
        """
        code is plot color
        :return: image
        """
        cv_rgb = cv2.cvtColor(self.image, code)
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(cv_rgb)
        fig.show()

    def card_type(self):
        """
        find card type
        :return: string
        """
        image = self.gray
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

    # modify: 2020.3.20
    #     def img_to_arr(self, x, y):
    #         """
    #         Convert image to array
    #         :param x: shape[0]
    #         :param y: shape[1]
    #         :return:
    #         """
    #         img = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)).convert('L')
    #         if img.size[0] != x or img.size[1] != y:
    #             img = img.resize((x, y))
    #
    #         arr = []
    #
    #         for i in range(y):
    #             for j in range(x):
    #                 pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
    #                 arr.append(pixel)
    #         return arr

    def format_input(self):
        """
        convert image into input data
        :return: np.array
        """
        input_img_text = np.array(img_to_arr(self.gray, 100, 100))
        vertex, img = find_form(self.gray)
        input_img_form = np.array(img_to_arr(img, 200, 200))

        input_img_text = input_img_text.reshape(1, 100, 100, 1)
        input_img_form = input_img_form.reshape(1, 200, 200, 1)

        return input_img_text, input_img_form

    def form_type(self):
        """
        Form classification for burial cards
        :return: string
        """
        """Eric adaption"""
        model_form = tf.keras.models.load_model(r'./Scripts/model_form.h5')
        # model_form = load_model(r'.\Scripts\model_form.h5')
        # update: 2020.3.20
        text_list, form_list = self.format_input()
        input_data = keras_input_data(form_list, 200, 200)
        ret = model_form.predict(input_data)
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
        """Eric adaption"""
        model_hw = tf.keras.models.load_model(r'./Scripts/model_hw.h5')
        # model_hw = load_model(r'.\Scripts\model_hw.h5')
        # update: 2020.3.20
        text_list, form_list = self.format_input()
        input_data = keras_input_data(text_list, 100, 100)
        ret = model_hw.predict(input_data)
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
        """Eric adaption"""
        model_f = tf.keras.models.load_model(r'./Scripts/model_f.h5')
        # model_f = load_model(r'.\Scripts\model_f.h5')
        # update: 2020.3.20
        text_list, form_list = self.format_input()
        input_data = keras_input_data(text_list, 100, 100)
        ret = model_f.predict(input_data)
        rel = np.where(ret[0] == np.max(ret[0]))
        if rel[0] == [0]:
            return 'N'
        else:
            return 'Y'

    def preproc_img(self):
        """
        update: 2020.3.20
        pre-process image:
        :return:
        """
        ret, th1 = cv2.threshold(self.gray, 200, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(th1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        # draw two vertical lines:
        h, w = th2.shape
        lines = [[23, 0, 23, h], [w - 50, 0, w - 50, h]]

        for l in lines:
            self.image = cv2.line(th2, (l[0], l[1]), (l[2], l[3]), (0, 0, 0), 2)

        return self.image

    def first_line(self):
        """
        update: 2020.3.20
        detect first line of image
        :return:
        """
        image = self.preproc_img()

        h, w = image.shape
        vertex, merge = find_form(image)
        ys, xs = np.where(vertex > 0)
        y_list = []
        for i in range(len(ys) - 1):
            if ys[i + 1] - ys[i] > 20:
                y_list.append(ys[i])
        # y_list.append(ys[i])
        # horizontal line
        horizontal_lines = []
        for y in y_list:
            horizontal_lines.append([0, y, w, y])
        if horizontal_lines and horizontal_lines[0][1] < 150:
            if horizontal_lines[0][1] > 50:
                first_line = horizontal_lines[0]
            else:
                try:
                    first_line = horizontal_lines[1]
                except IndexError:
                    first_line = [0, 104, w, 104]
        else:
            first_line = [0, 104, w, 104]

        return first_line


class CardDetect(CardOcr):
    """
    OCR for card recognition
    """

    # def __init__(self, img_path):
    #    super().__init__(img_path)
    #    self.image = cv2.imread(img_path, 1)
    """Adaption made by Eric"""

    def __init__(self, img):
        super().__init__(img)
        # self.image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def draw_auxiliary(self):
        """
        update: 2020.3.20
        draw auxiliary to crop the cell
        :return: image
        """
        # plot lines
        # proc_image = self.preproc_img()
        first_line = self.first_line()
        first_line[1] -= 55
        first_line[3] -= 55
        self.image = cv2.line(self.image, (first_line[0], first_line[1]), (first_line[2], first_line[3]), (0, 0, 255),
                              2)

        return self.image

    def draw_auxiliary_form_c(self):
        """
        draw lines for form C
        :return: image
        """
        first_line = self.first_line()
        # image = self.preproc_img()
        # image = self.draw_auxiliary()
        if first_line[1] > 100:
            first_line[1] -= 20
            first_line[3] -= 20
        self.image = cv2.line(self.image, (first_line[0], first_line[1]), (first_line[2], first_line[3]), (0, 0, 255),
                              2)
        new_line = [first_line[0], first_line[1] - 60, first_line[2], first_line[3] - 60]
        lines = [new_line, first_line]

        i = 0
        while i < 7:
            temp = first_line
            first_line[1] += 58
            first_line[3] += 58
            line = temp.copy()
            lines.append(line)
            i += 1

        for l in lines:
            self.image = cv2.line(self.image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2)

        return self.image

    def find_vertex(self):
        """
        update: 2020.3.20
        find vertex of cell
        :return: array, image
        """
        image = self.draw_auxiliary()
        vertex, merge = find_form(image)

        return vertex, merge

    # get the coordinate for Burial form A
    def get_coordinate_form_a(self):
        """
        get coordinate for burial form A
        :return: list
        """
        a, b = self.find_vertex()
        ys, xs = np.where(a > 0)
        # create coordinate
        x_list, y_list = [], []

        # sort list
        xs, ys = np.sort(xs), np.sort(ys)

        for i in range(len(xs) - 1):
            if xs[i + 1] - xs[i] > 20:
                x_list.append(xs[i])
        x_list.append(xs[i])
        if x_list[0] > 30:
            x_list.append(24)
        x_list = sorted(x_list)

        for i in range(len(ys) - 1):
            if ys[i + 1] - ys[i] > 20:
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

        return x_list, y_list[:3]

    def get_coordinate_other(self):
        a, b = self.find_vertex()
        ys, xs = np.where(a > 0)
        # create coordinate
        x_list, y_list = [], []

        # sort list
        xs, ys = np.sort(xs), np.sort(ys)

        for i in range(len(xs) - 1):
            if xs[i + 1] - xs[i] > 20:
                x_list.append(xs[i])
        x_list.append(xs[i])
        if x_list[0] > 30:
            x_list.append(24)
        x_list = sorted(x_list)

        for i in range(len(ys) - 1):
            if ys[i + 1] - ys[i] > 20:
                y_list.append(ys[i])
        y_list.append(ys[i])

        return x_list, y_list

    # find cell
    def cell_detect_form_a(self):
        """
        find cell for form a
        :return: list
        """
        x_list, y_list = self.get_coordinate_form_a()
        # print(x_list)
        rects = []

        try:
            first_line = [x_list[0], x_list[-3] - 25, x_list[-1]]
            for i in range(0, len(first_line) - 1):
                for j in range(len(y_list[:2]) - 1):
                    rects.append((first_line[i], y_list[j], first_line[i + 1], y_list[j + 1] - 20))
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
            self.image = cv2.rectangle(self.image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

        # print(rects)
        return rects

    def cell_detect_other(self):
        """
        find cell for other form except Form A
        :return: list
        """
        x_list, y_list = self.get_coordinate_other()
        rects = []

        try:
            for i in range(len(x_list) - 1):
                for j in range(len(y_list) - 1):
                    rects.append((x_list[i], y_list[j], x_list[i + 1], y_list[j + 1]))

        except IndexError:
            pass

        rect_ret = rects[:8]
        # plot rect:
        for rect in rect_ret:
            self.image = cv2.rectangle(self.image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

        # print(rects)
        return rect_ret

    # extract text:
    def ocr_text_form_a(self):
        rects = self.cell_detect_form_a()
        thresh = self.preproc_img()
        target = [0, 2, 3, 4, 5]
        file_name = ['Name', 'Date of interment', 'Section', 'Lot', 'GR']
        special_char = '‘’,|-_<"=;«“&—]uv(é*§¢!'
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
                    text1 = re.sub('\.0', "", text1)
                    text1 = re.sub("\s+", " ", ''.join(re.findall(r'[A-Za-z0-9]\s*', text1)))

                result[name] = text1
                # print(name, ':', text1, end='\n')
            result['Avg_conf'] = round((conf / 5), 2)
        except IndexError:
            pass

        return result

    def ocr_text_form_bc(self):
        rectangle = self.cell_detect_other()
        rects = rectangle[:5]
        image = self.preproc_img()
        # print(rects)
        file_name = ['Name', 'Lot-Sec-Gr-Ter', 'Date of Burial']
        special_char = '‘’,|-_<"=;«“&—]uv(é*§¢!'
        result = {}
        temp = []
        for i in range(4):
            try:
                if i == 0:
                    rect1 = rects[i]
                    # print(rect1)
                    detect_img = image[rect1[1]:rect1[3], rect1[0]:rect1[2]]
                    text = pytesseract.image_to_string(detect_img, config='-l eng --psm 7')
                    text = ''.join([char for char in text if char not in special_char])
                    temp.append(text)
                else:
                    rect1 = rects[i]
                    # print(rect1)
                    detect_img = image[rect1[1]:rect1[3], rect1[0]:rect1[2]]
                    text = pytesseract.image_to_string(detect_img, config='-l eng snum --psm 7')
                    text = ''.join([char for char in text if char not in special_char])
                    temp.append(text)
            except IndexError:
                pass

        result[file_name[0]] = temp[0:1]
        result[file_name[1]] = temp[1:3]
        result[file_name[2]] = temp[3:]

        return result

    def ocr_text_deed(self):
        rects = self.cell_detect_other()
        image = self.preproc_img()
        # print(rects)
        file_name = ['Name', 'Lot-Sec-Gr', 'Deed No. & Date', 'Comments']
        special_char = '‘’,|-_<"=;«“&—]uv(é*§¢!'
        result = {}
        temp = []
        for i in range(8):
            if i in range(1, 5):
                rect1 = rects[i]
                # print(rect1)
                detect_img = image[rect1[1]:rect1[3], rect1[0]:rect1[2]]
                text = pytesseract.image_to_string(detect_img, config='-l eng snum --psm 7')
                text = ''.join([char for char in text if char not in special_char])
                text = re.sub("\s+", " ", ''.join(re.findall(r'[A-Za-z0-9]\s*\/*', text)))
                temp.append(text)
            else:
                rect1 = rects[i]
                # print(rect1)
                detect_img = image[rect1[1]:rect1[3], rect1[0]:rect1[2]]
                text = pytesseract.image_to_string(detect_img, config='-l eng --psm 7')
                text = ''.join([char for char in text if char not in special_char])
                text = re.sub("\s+", " ", ''.join(re.findall(r'[A-Za-z0-9]\s*',
                                                             re.sub("[,|()|-]", " ", text)))).upper()
                temp.append(text)

        result[file_name[0]] = temp[0:1]
        result[file_name[1]] = temp[1:3]
        result[file_name[2]] = temp[3:5]
        result[file_name[3]] = temp[5:]

        return result


def cls_dict(image):
    image_info = {}
    card = CardDetect(image)
    image_info['file_name'] = image

    hw = card.flag_hw()
    image_info['handwriting'] = hw

    fraction = card.flag_f()
    image_info['fraction'] = fraction

    card_type = card.card_type()
    image_info['card_type'] = card_type

    form = card.form_type()
    image_info['form'] = form

    return image_info


def text_dict(image):
    card = CardDetect(image)
    card_type = card.card_type()
    form = card.form_type()
    if card_type == 'Burial' and form == 'A':
        image_text = card.ocr_text_form_a()
    elif card_type == 'Burial' and form != 'A':
        image_text = card.ocr_text_form_bc()
    else:
        image_text = card.ocr_text_deed()

    return image_text


def image_ocr(image):
    image_ret = {}
    card = CardDetect(image)

    hw = card.flag_hw()
    image_ret['handwriting'] = hw

    fraction = card.flag_f()
    image_ret['fraction'] = fraction

    # card_type = card.card_type()
    """Eric adaption"""
    # card_type = card.card_type().lower()
    """ update: 2020.3.20"""
    card_type = card.card_type()
    image_ret['card_type'] = card_type.lower()

    form = card.form_type()
    image_ret['form'] = form

    if card_type == 'Burial' and form == 'A':
        temp = card.ocr_text_form_a()
        image_ret.update(temp)
    elif card_type == 'Burial' and form != 'A':
        temp = card.ocr_text_form_bc()
        image_ret.update(temp)
    else:
        temp = card.ocr_text_deed()
        image_ret.update(temp)

    return image_ret


def ocr_forma_f(image):
    """
    Demo for form a with fraction, iCard_021886_1_Dace_Frank.jpg
    """
    image_ret = {}
    card = CardDetect(image)
    image_ret['file_name'] = image

    image_ret['handwriting'] = 'N'

    image_ret['fraction'] = 'Y'

    card_type = card.card_type()
    image_ret['card_type'] = card_type

    image_ret['form'] = 'A'

    temp = card.ocr_text_form_a()
    image_ret.update(temp)

    return image_ret


def ocr_forma(image):
    """
    Demo for form a without fraction: iCard_021875_1_Daba_Shorro.jpg
    """
    image_ret = {}
    card = CardDetect(image)

    image_ret['handwriting'] = 'N'

    image_ret['fraction'] = 'N'

    card_type = card.card_type()
    image_ret['card_type'] = card_type

    image_ret['form'] = 'A'

    temp = card.ocr_text_form_a()
    image_ret.update(temp)

    return image_ret


def main():
    """
    create dataframe
    :return: csv file
    """
    path = 'All_Data/'
    files = sorted(os.listdir(path))
    ret_burial = []
    ret_deed = []

    for i in files:
        card = CardDetect(path + i)
        if card.card_type() == 'Burial':
            card_info = cls_dict(path + i)
            ret_burial.append(card_info)
            text_info = text_dict(path + i)
            ret_burial.append(text_info)
        else:
            card_info = cls_dict(path + i)
            ret_deed.append(card_info)
            text_info = text_dict(path + i)
            ret_deed.append(text_info)

    df_burial = pd.DataFrame(ret_burial, columns=['file_name', 'handwriting', 'fraction', 'card_type', 'form_type',
                                                  'Name', 'Date of interment', 'Section', 'Lot', 'GR', 'Avg_conf'])
    df_deed = pd.DataFrame(ret_burial, columns=['file_name', 'handwriting', 'fraction', 'card_type', 'form_type',
                                                'Name', 'Lot-Sec-Gr', 'Deed No. & Date', 'Comments'])
    return df_burial.to_csv('result_burial.csv'), df_deed.to_csv('result_deed.csv')


if __name__ == '__main__':
    main()
