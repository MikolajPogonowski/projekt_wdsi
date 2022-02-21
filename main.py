
import os
import re
import shutil
import glob
import re
import cv2
import random
from xml.dom import minidom
import xml.etree.ElementTree as ElementTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#import PIL.Image as Image
import matplotlib.pyplot as plt
import lxml.etree as etree
import numpy as np

# TODO Jakość kodu i raport (4/4)


# TODO Skuteczność klasyfikacji 0.916 (4/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0

# TODO Skuteczność detekcji (0/2)

# TODO Poprawki po terminie. (-1)


path_main = os.path.abspath('main.py')
path_repo = os.path.dirname(path_main)
path_ok = os.path.dirname(path_repo)
data_dir = os.path.join(path_ok, "dataset")

dataset_annotations_xml_loc = os.path.join(data_dir, 'annotations', '*.xml')
limits_annotations_xml_loc = os.path.join(path_ok, 'limits', 'annotations', '*.xml')
others_annotations_xml_loc = os.path.join(path_ok, 'others', 'annotations', '*.xml')

def import_data(path, set):
    images_path = os.path.join(path, set, 'images/*.png')
    annotations_path = os.path.join(path, set, 'annotations/*.xml')
    images_list = glob.glob(images_path)
    annotations_list = glob.glob(annotations_path)
    images_list.sort()
    # for n in images_list:
    #     print(n)
    annotations_list.sort()
    # for n in annotations_list:
    #     print(n)
    data_list = []
    n = 0

    for k in annotations_list:  # przeglądanie kolejnych plików xml
        element = ElementTree.parse(k)  # 'otwiera xmla', jako element tree, znajdujemy się w sekcji xmla <annotations>
        root = element.getroot() # umożliwia dostęp do danych 'głębiej' w xmlu

        amount_all = 0
        amount_limits = 0

        width = int(root.find('size/width').text)  # pozyskanie szerokości zdjęcia
        height = int(root.find('size/height').text)  # pozyskanie wysokości zdjęcia
        # print(width, height)

        filename = root.find('filename').text

        obj = element.findall('object/name')
        dict = {}
        # TODO To mozna zrobic w petli ponizej.
        for i in obj:
            name = i.text

            if name in ['speedlimit']:
                flag = 1
                amount_limits += 1
                amount_all += 1

            else:
                amount_all +=1

            #print('\n')

        dict['image'] = images_list[n]  # dodawanie ściezki do obrazka
        dict['annotation'] = annotations_list[n]    # dodawanie ścieżki do xml
        dict['height'] = height
        dict['width'] = width
        dict['signs_number'] = amount_all
        dict['limits_number'] = amount_limits

        n += 1

        all_signs = root.findall('object')  # znajduje wszystkie znaki (objects) w obrębie JEDNEGO zdjęcia w xml

        # dodawanie współrzędnych ramki ze znakiem speedlimit:
        j = 1
        for s in all_signs:
            name = s.find('name').text
            sign_frame = []
            if name == 'speedlimit':
                dict[f'limit_flag{j}'] = 1
            else:
                dict[f'limit_flag{j}'] = 0

            xmin = int(s.find('bndbox/xmin').text)
            dict[f'x_{j}_min'] = xmin
            ymin = int(s.find('bndbox/ymin').text)

            dict[f'y_{j}_min'] = ymin
            xmax = int(s.find('bndbox/xmax').text)
            dict[f'x_{j}_max'] = xmax
            ymax = int(s.find('bndbox/ymax').text)
            dict[f'y_{j}_max'] = ymax


            j += 1


        data_list.append(dict)

    return data_list

def input_2():
    test_images_path = os.path.join(path_ok, 'test', 'images')
    #print(test_images_path)
    i = 0
    #print('Liczba plików do przetworzenia: ')
    n_files_str = input()
    n_files = int(n_files_str)
    data_input = []

    for n in range(n_files):
        #print('Nazwa ', i + 1, ' pliku: ')
        file_1 = input()
        img_path = os.path.join(test_images_path, file_1)
        img = cv2.imread(img_path)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)

        #print('Liczba wycinków obrazu do sklasyfikowania na ', i + 1, ' zdjęciu: ')
        n_1_str = input()
        n_1 = int(n_1_str)

        for k in range(n_1):
            dict = {}
            #print('Podaj współrzędne dla ramki ', k+1, ": xmin, xmax, ymin ymax")
            frame = input().split()
            frame_int = [int(x) for x in frame]
            #dict[f'frame{k+1}'] = frame_int
            xmin = frame_int[0]
            ymin = frame_int[2]
            xmax = frame_int[1]
            ymax = frame_int[3]
            crop_img = img[ymin:ymax, xmin:xmax]
            #cv2.imshow('crop', crop_img)
            #cv2.waitKey(0)
            dict['image'] = crop_img    # to trafi na sifta i bovw

            data_input.append(dict)
        i += 1


    #print(data_input)

    return data_input

def crop_images(database):
    cropped_signs = []
    for data in database:
        image_path = data['image']
        #print(image_path)
        dict = []


        for i in range(data['signs_number']):
            xmin = data[f'x_{i+1}_min']
            ymin = data[f'y_{i+1}_min']
            xmax = data[f'x_{i+1}_max']
            ymax = data[f'y_{i+1}_max']
            #print(xmin, ymin, xmax, ymax)
            img = cv2.imread(image_path)
            crop_img = img[ymin:ymax, xmin:xmax]
            #cv2.imshow("speedlimit", crop_img)
            #cv2.waitKey(0)

            if data[f'limit_flag{i+1}'] == 1:
                label = 1
            else:
                label = 0

            # print(image_path)
            # print(label)

            cropped_signs.append({'image': crop_img, 'label': label})





    return cropped_signs

def learn_bovw(data):
    """
    Learns BoVW dictionary and saves it as "voc.npy" file.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    @return: Nothing
    """
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)

def extract_features(data):
    """
    Extracts features for given data and saves it as "desc" entry.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    @return: Data with added descriptors for each sample.
    """
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)
        sample['desc'] = desc

    return data

def train(data):
    """
    Trains Random Forest classifier.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Trained model.
    """
    descs = []
    labels = []
    for sample in data:
        if sample['desc'] is not None:
            descs.append(sample['desc'].squeeze(0))
            labels.append(sample['label'])
    rf = RandomForestClassifier()
    rf.fit(descs, labels)

    return rf

def predict(rf, data):
    """
    Predicts labels given a model and saves them as "label_pred" (int) entry for each sample.
    @param rf: Trained model.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Data with added predicted labels for each sample.
    """

    for sample in data:
        flag = 0
        if sample['desc'] is not None:
            predict = rf.predict(sample['desc'])
            #print(predict)
            flag = int(predict)
            sample['label_pred'] = int(predict)
            #print(111)

        ###
        if flag == 1:
            print('speedlimit')
        else:
            print('other')
        ###


    # ------------------
    return data

def evaluate(data):
    """
    Evaluates results of classification.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor), and "label_pred".
    @return: Nothing.
    """
    # evaluate classification results and print statistics
    # TODO PUT YOUR CODE HERE
    correct = 0
    incorrect = 0
    eval = []
    real = []
    for sample in data:
        if sample['desc'] is not None:
            eval.append(sample['label_pred'])
            real.append(sample['label'])
            if sample['label_pred'] == sample['label']:
                correct += 1
            else:
                incorrect += 1

    print('score = %.3f' % (correct / (correct + incorrect)))

    conf = confusion_matrix(real, eval)
    print(conf)
    # ------------------

    # this function does not return anything
    return

def draw_grid(images, n_classes, grid_size, h, w):
    """
    Draws images on a grid, with columns corresponding to classes.
    @param images: Dictionary with images in a form of (class_id, list of np.array images).
    @param n_classes: Number of classes.
    @param grid_size: Number of samples per class.
    @param h: Height in pixels.
    @param w: Width in pixels.
    @return: Rendered image
    """
    image_all = np.zeros((h, w, 3), dtype=np.uint8)
    h_size = int(h / grid_size)
    w_size = int(w / n_classes)

    col = 0
    for class_id, class_images in images.items():
        for idx, cur_image in enumerate(class_images):
            row = idx

            if col < n_classes and row < grid_size:
                image_resized = cv2.resize(cur_image, (w_size, h_size))
                image_all[row * h_size: (row + 1) * h_size, col * w_size: (col + 1) * w_size, :] = image_resized

        col += 1

    return image_all

def display(data):
    """
    Displays samples of correct and incorrect classification.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor), and "label_pred".
    @return: Nothing.
    """
    n_classes = 3

    corr = {}
    incorr = {}

    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            if sample['label_pred'] == sample['label']:
                if sample['label_pred'] not in corr:
                    corr[sample['label_pred']] = []
                corr[sample['label_pred']].append(idx)
            else:
                if sample['label_pred'] not in incorr:
                    incorr[sample['label_pred']] = []
                incorr[sample['label_pred']].append(idx)

            # print('ground truth = %s, predicted = %s' % (sample['label'], pred))
            # cv2.imshow('image', sample['image'])
            # cv2.waitKey()

    grid_size = 8

    # sort according to classes
    corr = dict(sorted(corr.items(), key=lambda item: item[0]))
    corr_disp = {}
    for key, samples in corr.items():
        idxs = random.sample(samples, min(grid_size, len(samples)))
        corr_disp[key] = [data[idx]['image'] for idx in idxs]
    # sort according to classes
    incorr = dict(sorted(incorr.items(), key=lambda item: item[0]))
    incorr_disp = {}
    for key, samples in incorr.items():
        idxs = random.sample(samples, min(grid_size, len(samples)))
        incorr_disp[key] = [data[idx]['image'] for idx in idxs]

    image_corr = draw_grid(corr_disp, n_classes, grid_size, 800, 600)
    image_incorr = draw_grid(incorr_disp, n_classes, grid_size, 800, 600)

    cv2.imshow('images correct', image_corr)
    cv2.imshow('images incorrect', image_incorr)
    cv2.waitKey()

    # this function does not return anything
    return

def main():
# training start

    #print('loading train data...')
    imported_train_data = import_data(path_ok, 'train')
    # for j in imported_train_data:
    #     print(j)

    data_train = crop_images(imported_train_data)
    #print('train data loaded')

    #print('learning_bovw...')
    learn_bovw(data_train)
    #print('bowv_learned')

    #print('extracting_features...')
    cropped_signs_1 = extract_features(data_train)
    #print('features_extracted')

    #print('training_model...')
    rf = train(cropped_signs_1)
    #print("model_trained")

# training end

# input start
    #print('Type "classify": ')
    operation = input()
    #operation = 'classify'
    if operation == 'classify':
        #input_data = input_2()
        data_test = input_2()

        #print('testing...')
        data_test = extract_features(data_test)
        predict(rf, data_test)

        #evaluate(data_test)
        #display(data_test)
        #print('tested')

if __name__ == '__main__':
    main()


