# -*- coding: utf-8 -*-

# для создания случайных величин
import random

# Импортируем tensorflow
# import tensorflow as tf

# бэкенд Keras
# import tensorflow.keras.backend as K

# Модули конвертации между RGB и HSV
# from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# Модули работы с изображениями
from PIL import Image, ImageDraw, ImageFont

# import struct
# import gdown

# библиотека numpy
import numpy as np

# Слои нейронной сети
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Lambda

# Оптимизатор Adam
from tensorflow.keras.optimizers import Adam

# работа со слоями
from tensorflow.keras.layers import add, concatenate

# создание моделей
from tensorflow.keras.models import Model

# загрузчик сохраненных моделей
from tensorflow.keras.models import load_model

# итератор, повторно возвращающий указанный объект
from itertools import repeat

# регуляризатор, который применяет штраф за регуляризацию L2
from tensorflow.keras.regularizers import l2

# модуль для отрисовки изображения
from tensorflow.keras.preprocessing import image

# Импортируем tensorflow
import tensorflow as tf

# прямоугольник, определяемый точкой привязки xy , а также его шириной и высотой
# from matplotlib.patches import Rectangle

# возвращение списка (возможно, пустого) путей, соответствующих шаблону pathname
from glob import glob

# библиотека для работы с файлами
import os

def rand(a=0, b=1):

    return np.random.rand()*(b-a) + a

def DBL(x, filters, kernel, strides=1, batch_norm=True, layer_idx=None): # DarknetConv2D_BN_Leaky
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # Делаем oтступ в виде нулей по контуру изображения, что бы захватить левый верхний угол
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=kernel,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005), name='conv_' + str(layer_idx))(x)
    if batch_norm:
        x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(layer_idx))(x)
        x = LeakyReLU(alpha=0.1,name='leake_' + str(layer_idx))(x)
    return x, layer_idx+1

# Определяем минимальную ячейку Residual блока
def Res_unit(x, filters, layer_idx): # DarknetResidual
    skip_connection = x
    x, layer_idx = DBL(x, filters // 2, kernel=1, layer_idx=layer_idx)
    x, layer_idx = DBL(x, filters, kernel=3, layer_idx=layer_idx)
    x = add([skip_connection , x], name='Add_'+str(layer_idx))
    return x, layer_idx+1

# Определяем Residual блок состоящий из входного сверточного слоя и последовательности Res_unit блоков
def ResBlock(x, filters, blocks, layer_idx): # DarknetBlock
    x, layer_idx = DBL(x, filters, kernel=3, strides=2, layer_idx=layer_idx)
    for _ in repeat(None, blocks):
        x, layer_idx = Res_unit(x, filters, layer_idx=layer_idx)
    return x, layer_idx

def Detector(x_in, filters, layer_idx=None, num_sub_anchors=3, num_classes=1):
    if isinstance(x_in, list): # Если на вход поступает список попадаем в эту ветку (маршруты 2 и 3)
        x, x_skip = x_in[0], x_in[1]# Разбиваем список на отдельные тензоры
        x,layer_idx = DBL(x, filters, kernel=1, strides=1, layer_idx=layer_idx) # DarknetConv
        x = UpSampling2D(2, name = 'UpSampling_' + str(layer_idx))(x) # Повышаем размерность тензора
        layer_idx+=1
        x =concatenate([x, x_skip], name = 'Concatenate_' + str(layer_idx)) # Объединяем маршруты
        layer_idx+=1
        # Пять сверточных слоев DBL*5
        for i in range(2):
            x, layer_idx = DBL(x, filters, 1, layer_idx=layer_idx)      # 1,3
            x, layer_idx = DBL(x, filters * 2, 3, layer_idx=layer_idx)  # 2,4
        fork, layer_idx = DBL(x, filters, 1, layer_idx=layer_idx)      # 5 С пятого слоя каскада делаем вилку на выход и на другой масштаб
    else: # В эту ветку попадает только маршрут 1
        x = x_in
        # Пять сверточных слоев DBL*5
        for i in range(2):
            x, layer_idx = DBL(x, filters, 1, layer_idx=layer_idx)      # 1,3
            x, layer_idx = DBL(x, filters * 2, 3, layer_idx=layer_idx)  # 2,4
        fork, layer_idx = DBL(x, filters, 1, layer_idx=layer_idx)      # 5 С пятого слоя каскада делаем вилку на выход и на другой масштаб

    # Выходные слои
    x,layer_idx = DBL(fork, filters=filters*2, kernel=3, strides=1, layer_idx=layer_idx)    # Предпоследний сверточный слой (№80 13х13х1024, #92 26x26x512, #104 52x52x256)
    bboxes, layer_idx = DBL(x, filters=num_sub_anchors * (5+num_classes), kernel=1, strides=1, batch_norm= False, layer_idx=layer_idx)# Выходные слои (№81 13х13х (anchors * (5 + classes)), №93 26х26, №105 52х52 (255)

    return bboxes, fork, layer_idx

def create_yolov3_model(inputs, num_sub_anchors, num_classes):

    # Так бы выглядела Сеть Darknet 53 если бы ее использовали отдельно
    layer_idx = 0 # Номер первого слоя
    x, layer_idx = DBL(inputs, filters=32, kernel=3, layer_idx=layer_idx)       # DarknetConv 1 слой
    x, layer_idx = ResBlock(x, filters=64, blocks=1, layer_idx=layer_idx)            # DarknetBlock 3 слоя
    x, layer_idx = ResBlock(x, filters=128, blocks=2, layer_idx=layer_idx)           # DarknetBlock 5 слоя
    x, layer_idx = Route_1,_ = ResBlock(x, filters=256, blocks=8, layer_idx=layer_idx) # DarknetBlock 9 слоев
    x, layer_idx = Route_2,_ = ResBlock(x, filters=512, blocks=8, layer_idx=layer_idx) # DarknetBlock 9 слоев
    Route_3, layer_idx = ResBlock(x, filters=1024, blocks=4, layer_idx=layer_idx)          # последние 4 Res блока Darknet

    bbox_scale_1, fork_1, layer_idx = Detector(Route_3, filters=512, layer_idx=layer_idx, num_sub_anchors=num_sub_anchors, num_classes=num_classes) # 5 сверточных слоев DBL

    # 82 слой на первый выход  83 пропуск
    layer_idx = 84
    bbox_scale_2, fork_2, layer_idx = Detector([fork_1, Route_2], filters=256, layer_idx=layer_idx, num_sub_anchors=num_sub_anchors, num_classes=num_classes) # 6 слоев

    # слои 94-95 пропущены
    layer_idx = 96
    bbox_scale_3, _, layer_idx = Detector([fork_2, Route_1], filters=128, layer_idx=layer_idx, num_sub_anchors=num_sub_anchors, num_classes=num_classes) # 6 слоев

    model = Model (inputs, [bbox_scale_1, bbox_scale_2, bbox_scale_3])
    return model


## Функция для обнаружения объект

def object_detection (image, model_YOLO, probability=0.7, x_size=12, y_size=17):
    name_classes = ['Medic']
    num_classes = len(name_classes)
    anchors = np.array([[10,13], [16,30], [33,23], [30, 61], [62,45], [59,119], [116, 90], [156, 198], [373, 326]])
    # Создаем набор цветов для ограничивающих рамок
    import colorsys
    hsv_tuples = [(x / len(name_classes), 1., 1.) for x in range(len(name_classes))]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # colors = list(map(lambda x:(int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
    colors = [(250, 223, 134)]
    # np.random.seed(43)
    # np.random.shuffle(colors)
    # np.random.seed(None)

    # Изменяем размер картинки под input_shape
    iw, ih = image.size
    w, h = (416, 416)
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image_for_predict = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (416,416), (128, 128, 128))
    new_image.paste(image_for_predict, ((w - nw) // 2, (h - nh) // 2))
    image_for_predict = new_image
    image_for_predict = np.array(image_for_predict) / 255.
    image_for_predict = image_for_predict.reshape(1, 416, 416, 3)

    predict = model_YOLO.predict(image_for_predict)
    print('predict', predict)
    num_layers = len(predict) # Получаем количество сеток
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] # Задаем маски для 3 уровней анкоров
    input_shape = np.array(predict[0].shape[1:3]) * 32  # Получаем размер выходного изображения
    image_shape = np.array([image.size[1], image.size[0]]) # Сохраняем размер оригинального изображения

    level_anchor = 0 # Укажем уровень сетки
    num_anchors = len(anchors[anchor_mask[level_anchor]]) # Получаем количество анкоров
    anchors_tensor = np.reshape(anchors[anchor_mask[level_anchor]], (1,1,1,num_anchors,2)) # Выбираем анкоры для нашего уровня сетки и решейпим

    # Создаем пустую сетку
    grid_shape = predict[level_anchor].shape[1:3] # Получим размерность сетки
    grid = [] # Массив для финальной сетки
    grid_row = [] # Массив для столбца
    for i in range(grid_shape[0]): # По всем строкам
        for j in range(grid_shape[1]): # По всем столбцам
            grid_row.append([j , i]) # Создаем элемент [j, i]
        grid.append(grid_row) # Добавляем столбец в финальную сетку
        grid_row = [] # Обнуляем данные для столбца
    grid = np.array(grid) # Переводим в numpy
    grid = np.expand_dims(grid, axis=2) # Добавляем размерность

    # Функция расчета сигмоиды для вектора
    def sigmoid(x): # На вход подаем массив данных
        return 1/(1+np.exp(-x)) # Возвращаем сигмоиду для всех элементов массива

    # Решейпим предикт
    feats = np.reshape(predict[level_anchor], (-1, grid_shape[0], grid_shape[1], num_anchors, num_classes+5))

    # Координаты центра bounding box
    xy_param = feats[..., :2] # Выцепляем 0 и 1 параметры из предикта (соответствуют параметрам смещения центра анкора)
    box_xy = (sigmoid(xy_param) + grid)/grid_shape[::-1] # Получаем координаты центра bounding box

    # Высота и ширна bounding box
    wh_param = feats[..., 2:4] # Выцепляем 2 и 3 параметры из предикта (соответствуют праметрам изменения высоты и ширины анкора)
    box_wh = np.exp(wh_param) * anchors_tensor / input_shape[::-1] # Получаем высоту и ширину bounding box

    # Вероятность наличия объекта в анкоре
    conf_param = feats[..., 4:5] # Выцепляем 4 параметр из предикта (соответствуют вероятности обнаружения объекта)
    box_confidence = sigmoid(conf_param) # Получаем вероятность наличия объекта в bounding box

    # Класс объекта
    class_param = feats[...,5:] # Выцепляем 5+ параметры из предикта (соответствуют вероятностям классов объектов)
    box_class_probs = sigmoid(class_param) # Получаем вероятности классов объектов

    # Корректируем ограничивающие рамки (Размер изображения на выходе 416х416)
    # И найденные параметры соответствуют именно этой размерности
    # Необходимо найти координаты bounding box для рамерности исходного изображения
    box_yx = box_xy[..., ::-1].copy()
    box_hw = box_wh[..., ::-1].copy()

    new_shape = np.round(image_shape * np.min(input_shape/image_shape)) # Находим размерность пропорциональную исходной с одной из сторон 416
    offset = (input_shape-new_shape)/2./input_shape # Смотрим на сколько надо сместить в относительных координатах
    scale = input_shape/new_shape  # Находим коэфициент масштабирования
    box_yx = (box_yx - offset) * scale # Смещаем по координатам
    box_hw *= scale # Масштабируем ширину и высоту

    box_mins = box_yx - (box_hw / 2.) # Получаем левые верхние координаты (от середины отнимаем половину ширины и высоты)
    box_maxes = box_yx + (box_hw / 2.) # Получаем правые нижнние координаты (к середине прибавляем половину ширины и высоты)
    _boxes =  np.concatenate([
        box_mins[..., 0:1], # yMin
        box_mins[..., 1:2], # xMin
        box_maxes[..., 0:1], # yMax
        box_maxes[..., 1:2]  # xMax
    ], axis=-1)

    _boxes *= np.concatenate([image_shape, image_shape]) # Переводим из относительных координат в абсолютные

    # Получаем выходные параметры
    _boxes_reshape = np.reshape(_boxes, (-1, 4)) # Решейпим все боксы в один массив
    _box_scores = box_confidence * box_class_probs # Получаем вероятность каждого класса (умноженную на веоятность наличия объекта)
    _box_scores_reshape = np.reshape(_box_scores, (-1, num_classes)) # Решейпим в один массив

    mask = _box_scores_reshape >= probability # Берем все объекты, обнаруженные с вероятностью больше 0.7
    _boxes_out = _boxes_reshape[mask[:,0]]
    _scores_out = _box_scores_reshape[:, 0][mask[:,0]]
    classes_out = np.ones_like(_scores_out,'int32') * 0
    # font = ImageFont.truetype(font=path + 'font.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    font = ImageFont.truetype(font='resource/font.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    image_pred = image.copy()
    print('classes_out', classes_out)
    for i, c in reversed(list(enumerate(classes_out))):
        print(c)
        draw = ImageDraw.Draw(image_pred)
        predicted_class = name_classes[c]
        box = _boxes_out[i]
        score = _scores_out[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image_pred
    # plt.figure(figsize=(x_size,y_size))
    # plt.imshow(image_pred)

"""## Тестирование модели на собственных весах

Создаем модель
"""

yolo3 = create_yolov3_model(Input(shape=(416, 416, 3)), 3, 1)

#gdown.download('https://storage.yandexcloud.net/aiueducation/Content/advanced/l9/yolo_new.h5', None, quiet=True)

yolo3.load_weights('model/yolo_new.h5')
