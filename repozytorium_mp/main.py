
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

path_main = os.path.abspath('main.py')
path_repo = os.path.dirname(path_main)
path_ok = os.path.dirname(path_repo)
data_dir = os.path.join(path_ok, "dataset")
# print(data_dir)

dataset_annotations_xml_loc = os.path.join(data_dir, 'annotations', '*.xml')
limits_annotations_xml_loc = os.path.join(path_ok, 'limits', 'annotations', '*.xml')
others_annotations_xml_loc = os.path.join(path_ok, 'others', 'annotations', '*.xml')
# print(dataset_annotations_xml_loc)
# print(limits_annotations_xml_loc)
# print(others_annotations_xml_loc)

# def copy_data(loc_ori, loc_target, filename, filename_xml):
#     original = os.path.join(path_ok, loc_ori, "images", filename)
#     target = os.path.join(path_ok, loc_target, "images", filename)
#     shutil.copyfile(original, target)
#
#     original_xml = os.path.join(path_ok, loc_ori, "annotations", filename_xml)
#     target_xml = os.path.join(path_ok, loc_target, "annotations", filename_xml)
#     shutil.copyfile(original_xml, target_xml)
#
# def sort_data():
#     # path = os.path.abspath('main.py')
#     # p = os.path.dirname(path)
#     # p1 = os.path.dirname(p)
#     # print(p)
#     # print(p1)
#     # data_dir = os.path.join(p1, "dataset")
#     #annotations_dir = os.path.join(data_dir, "annotations")
#     #print(annotations_dir)
#
#     #data_dir = "dataset"
#     #annotations_dir = os.path.join(data_dir, "annotations")
#     #images_dir = os.path.join(data_dir, "images")
#     # annotations_files = os.listdir(annotations_dir) #ładuje wszystkie nazwy plików z folderu do listy
#     # #file_name = annotations_files[1]
#     # #full_file = os.path.join('dataset','annotations', file_name)
#
#
#
#     for k in glob.glob(dataset_annotations_xml_loc):  #przeglądanie kolejnych plików xml
#         element = ElementTree.parse(k)
#         #print(element)
#         flag = 0
#
#         obj = element.findall('filename')
#         for j in obj:
#             filename = j.text   #uzyskujemy nazwę obecnie sprawdzanego pliku png
#             filename_xml = re.sub('.png', '.xml', filename)     #uzyskujemy nazwę pliku xml
#             print(filename)
#
#         obj = element.findall('object/name')
#         for i in obj:
#             name = i.text
#             print(name)
#
#             if name in ['speedlimit']:
#                 flag = 1
#
#         if (flag == 1):
#             copy_data('dataset', 'limits', filename, filename_xml)
#         else:
#             copy_data('dataset', 'others', filename, filename_xml)
#
#         print(flag)
#         print('********************************')
#
# def train_or_test_limits():
#     n = 0
#     for k in glob.glob(limits_annotations_xml_loc):  #przeglądanie kolejnych plików xml
#         temp = ElementTree.parse(k)
#         n += 1
#
#         obj = temp.findall('filename')
#         for j in obj:
#             filename = j.text  # uzyskujemy nazwę obecnie sprawdzanego pliku png
#             filename_xml = re.sub('.png', '.xml', filename)  # uzyskujemy nazwę pliku xml
#             #print(filename)
#
#         if (n % 4 == 0):
#             copy_data('limits', 'test', filename, filename_xml)
#         else:
#             copy_data('limits', 'train', filename, filename_xml)
#
# def train_or_test_others():
#     n = 0
#     for k in glob.glob(others_annotations_xml_loc):  # przeglądanie kolejnych plików xml
#         temp = ElementTree.parse(k)
#         n += 1
#
#         obj = temp.findall('filename')
#         for j in obj:
#             filename = j.text  # uzyskujemy nazwę obecnie sprawdzanego pliku png
#             filename_xml = re.sub('.png', '.xml', filename)  # uzyskujemy nazwę pliku xml
#             # print(filename)
#
#         if (n % 4 == 0):
#             copy_data('others', 'test', filename, filename_xml)
#         else:
#             copy_data('others', 'train', filename, filename_xml)


# def input():
#     print('Type classify: ')
#     operation = input()
#     i=0
#     #if operation == 'classify':
#     if operation == 'c':
#         print('Liczba plików do przetworzenia: ')
#         n_files_str = input()
#         n_files = int(n_files_str)
#         file_names = []
#         frame_numbers = []
#
#         for n in range(n_files):
#             print('Nazwa ', i+1, ' pliku: ')
#             file_1 = input()
#             file_names.append(file_1)
#             print('Liczba wycinków obrazu do sklasyfikowania na ', i+1, ' zdjęciu: ')
#             n_1 = input()
#             frame_numbers.append(n_1)
#
#             # TODO - for do dodawania współrzędnych wszystkich sprawdzanych ramek
#
#             i += 1
#
#     # for x in range(len(frame_numbers)):
#     #     print(frame_numbers[x])

def import_data(path, set):
    images_path = os.path.join(path, set,'images/*.png')
    annotations_path = os.path.join(path, set,'annotations/*.xml')
    images_list = glob.glob(images_path)
    annotations_list = glob.glob(annotations_path)
    data_list = []
    data = []
    n = 0

    for k in annotations_list:  # przeglądanie kolejnych plików xml
        element = ElementTree.parse(k)  # 'otwiera xmla', jako element tree, znajdujemy się w sekcji xmla <annotations>
        root = element.getroot() # umożliwia dostęp do danych 'głębiej' w xmlu
        #print(element)
        flag = 0
        # obj = element.findall('filename')
        # for j in obj:
        #     filename_png = j.text  # uzyskujemy nazwę obecnie sprawdzanego pliku png
        #     filename_xml = re.sub('.png', '.xml', filename_png)  # uzyskujemy nazwę pliku xml
        #     print(filename_png)

        amount_all = 0
        amount_limits = 0

        filename = root.find('filename').text
        width = int(root.find('size/width').text)  # pozyskanie szerokości zdjęcia
        height = int(root.find('size/height').text)  # pozyskanie wysokości zdjęcia
        # print(width, height)

        filename = root.find('filename').text

        obj = element.findall('object/name')
        dict = {}
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

        # dodawanie labela - czy na zdjęciu występuje speedlimit
        # if (flag == 1):
        #     dict['limit_label'] = 1
        #     #data_list.append({'image': images_list[n], 'annotation': annotations_list[n], 'label': 1, 'signs_number': amount_all, 'limits_number': amount_limits})
        # else:
        #     dict['limit_label'] = 0
        #     #data_list.append({'image': images_list[n], 'annotation': annotations_list[n], 'label': 0, 'signs_number': amount_all, 'limits_number': amount_limits})

        n += 1

        all_signs = root.findall('object')  # znajduje wszystkie znaki (objects) w obrębie JEDNEGO zdjęcia w xml
        #print(filename)

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

            # # print(xmin, ymin, xmax, ymax)
            # img = cv2.imread(image_path)
            # crop_img = img[ymin:ymax, xmin:xmax]
            # # cv2.imshow("speedlimit", crop_img)
            # # cv2.waitKey(0)

            j += 1

                #print(name, xmin, ymin, xmax, ymax)

        data_list.append(dict)

    # for j in data_list:
    #     print(j['annotation'])

    return data_list

def input_data():
    i = 0
    print('Liczba plików do przetworzenia: ')
    n_files_str = input()
    n_files = int(n_files_str)
    data_input = []
    file_names = []
    frame_numbers = []

    for n in range(n_files):
        print('Nazwa ', i + 1, ' pliku: ')
        file_1 = input()
        #file_names.append(file_1)
        print('Liczba wycinków obrazu do sklasyfikowania na ', i + 1, ' zdjęciu: ')
        n_1_str = input()
        n_1 = int(n_1_str)

        dict = {}
        dict['file'] = file_1   #przypisanie nazwy do odpowiedniego pola w słowniku
        dict['frames_number'] = n_1     #przypisanie liczby sprawdzanych ramek w danym pliku do odpowiedniego pola w słowniku


        # print(dict('frames_number'))

        # for k in range(n_1):
        #     print('Podaj współrzędne dla ramki ', k+1, ": xmin, xmax, ymin, ymax")
        #     frame = input().split()
        #     frame_int = [int(x) for x in frame]
        #     dict[f'frame{k+1}'] = frame_int
        #     print(dict)
        #
        # i += 1



    data_input.append(dict)
    print(data_input)


    return 0

def input_2():
    test_images_path = os.path.join(path_ok, 'test', 'images')
    #print(test_images_path)
    i = 0
    print('Liczba plików do przetworzenia: ')
    n_files_str = input()
    n_files = int(n_files_str)
    data_input = []
    file_names = []
    frame_numbers = []

    for n in range(n_files):
        print('Nazwa ', i + 1, ' pliku: ')
        file_1 = input()
        #file_1 = 'road329.png' ####################################### to usunąć po testowaniu i zostawić linijkę powyżej
        # file_names.append(file_1)
        img_path = os.path.join(test_images_path, file_1)
        img = cv2.imread(img_path)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)

        print('Liczba wycinków obrazu do sklasyfikowania na ', i + 1, ' zdjęciu: ')
        n_1_str = input()
        n_1 = int(n_1_str)


        #dict = {}
        #dict['file'] = file_1  # przypisanie nazwy do odpowiedniego pola w słowniku
        #dict['frames_number'] = n_1  # przypisanie liczby sprawdzanych ramek w danym pliku do odpowiedniego pola w słowniku

        #print(dict('frames_number'))

        for k in range(n_1):
            dict = {}
            print('Podaj współrzędne dla ramki ', k+1, ": xmin, xmax, ymin ymax") # !!! tu trzeba przerobić na wpisywanie w innej kolejności, tak jak w instrukcji do projektu
            frame = input().split()
            frame_int = [int(x) for x in frame]
            #dict[f'frame{k+1}'] = frame_int
            xmin = frame_int[0]
            ymin = frame_int[2]
            xmax = frame_int[1]
            ymax = frame_int[3]
            crop_img = img[ymin:ymax, xmin:xmax]
            cv2.imshow('crop', crop_img)
            cv2.waitKey(0)
            dict['image'] = crop_img    # to trafi na sifta i bovw
            #print('Podaj label (1 - limit, 0 - other)')
            #label = int(input())
            #dict['label'] = label
            #print(dict)

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

        # todo - dodawanie losowych ramek do zbioru




    return cropped_signs

def sliding_window(image, stepSize, windowSize):    # windowSize = (width, height)
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # xuất ra từng cửa sổ
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def crop_others(database):
     cropped_others = []
    # for data in database:
    #     image_path = data['image']
    #     print(image_path)
    #
    #     # współrzędne losowanej ramki (źle, bo czasem losuje xmax takie samo jak xmin i się wywala)!!!
    #     xmin_o = random.randint(0, data['width'])
    #     #x1 = xmin_o + 10
    #     xmax_o = random.randint(xmin_o, data['width'])
    #     ymin_o = random.randint(0, data['height'])
    #     #y1 = ymin_o + 10
    #     ymax_o = random.randint(ymin_o, data['height'])
    #     #print(value)
    #
    #     img = cv2.imread(image_path)
    #     crop_img = img[ymin_o:ymax_o, xmin_o:xmax_o]
    #     #cv2.imshow("speedlimit", crop_img)
    #     #cv2.waitKey(0)
    #     cropped_others.append(crop_img)

     return cropped_others

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
    # perform prediction using trained model and add results as "label_pred" (int) entry in sample

    # for sample in data:
    # sample.update({'prediction':rf.predict(sample['desc'])[0]})
    for sample in data:
        if sample['desc'] is not None:
            predict = rf.predict(sample['desc'])
            #print(predict)
            sample['label_pred'] = int(predict)

            ###
            if int(predict) == 1:
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

    print('loading train data...')
    imported_train_data = import_data(path_ok, 'train')
    # for j in imported_train_data:
    #     print(j)

    data_train = crop_images(imported_train_data)
    print('train data loaded')

    # i = 0
    # x = 0
    # for j in data_train:
    # cv2.imshow("speedlimit", j['image'])
    # cv2.waitKey(0)
    # print(j['label'])
    #     x += int(j['label'])
    #     i += 1
    # print(i, x)

    print('learning_bovw...')
    learn_bovw(data_train)
    print('bowv_learned')

    print('extracting_features...')
    cropped_signs_1 = extract_features(data_train)
    print('features_extracted')

    print('training_model...')
    rf = train(cropped_signs_1)
    print("model_trained")

# training end

# input start
    print('Type "classify": ')
    operation = input()
    #operation = 'classify'
    if operation == 'classify':
        #input_data = input_2()
        data_test = input_2()
        # for j in data_test:
        #     print(j['label'])

# input end



# testing start

        #imported_test_data = import_data(path_ok, 'test')
        #data_test = crop_signs(imported_test_data)

        #print('testing...')
        data_test = extract_features(data_test)
        data_test = predict(rf, data_test)
        #evaluate(data_test)
        #display(data_test)
        #print('tested')

#testing end


if __name__ == '__main__':
    main()


