
import os
import re
import shutil
import glob
import re
import cv2
from xml.dom import minidom
import xml.etree.ElementTree as ElementTree
#import PIL.Image as Image
import matplotlib.pyplot as plt
import lxml.etree as etree

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


def input():
    operation = input("Type classify: ")
    i=0
    #if operation == 'classify':
    if operation == 'c':
        n_files_str = input('Liczba plików do przetworzenia: ')
        n_files = int(n_files_str)
        file_names = []
        frame_numbers = []

        for n in range(n_files):
            print('Nazwa ', i+1, ' pliku: ')
            file_1 = input()
            file_names.append(file_1)
            print('Liczba wycinków obrazu do sklasyfikowania na ', i+1, ' zdjęciu: ')
            n_1 = input()
            frame_numbers.append(n_1)

            # TODO - for do dodawania współrzędnych wszystkich sprawdzanych ramek

            i += 1

    for x in range(len(frame_numbers)):
        print(frame_numbers[x])

def import_data_train(path, set):
    # path_main = os.path.abspath('main.py')
    # path_rep = os.path.dirname(path_main)
    # path_ok = os.path.dirname(path_rep)
    #path_load = os.path.abspath(path, set)

    images_path = os.path.join(path, set,'images/*.png')
    annotations_path = os.path.join(path, set,'annotations/*.xml')
    #print(images_path)
    #print(annotations_path)

    images_list = glob.glob(images_path)
    annotations_list = glob.glob(annotations_path)
    #print(images_list)
    #print(annotations_list)

    data_paths = []
    data = []
    n = 0

    for k in annotations_list:  # przeglądanie kolejnych plików xml

        # data_paths.append({'image': images_list[n], 'annotation': annotations_list[n]})

        element = ElementTree.parse(k)
        # print(element)
        # print(annotations_list[n])
        # print(images_list[n])

        flag = 0

        # obj = element.findall('filename')
        # for j in obj:
        #     filename_png = j.text  # uzyskujemy nazwę obecnie sprawdzanego pliku png
        #     filename_xml = re.sub('.png', '.xml', filename_png)  # uzyskujemy nazwę pliku xml
        #     print(filename_png)

        obj = element.findall('object/name')
        for i in obj:
            name = i.text
            #print(name)

            if name in ['speedlimit']:
                flag = 1

        if (flag == 1):
            #print('spoko')
            data_paths.append({'image': images_list[n], 'annotation': annotations_list[n], 'label': 1})
        else:
            #print('niespoko')
            data_paths.append({'image': images_list[n], 'annotation': annotations_list[n], 'label': 0})

        n += 1

    # for j in data_paths:
    #     print(j['annotation'])

    return data_paths


if __name__ == '__main__':
    #input()
    data_train_paths = import_data_train(path_ok, 'train')

    for j in data_train_paths:
        print(j['image'])
        # testowanie cv2.imread
        path_1 = j['image']
        img = cv2.imread(path_1)
        cv2.imshow('image', img)
        cv2.waitKey()


    # slownik = []
    # slownik.append({'image': 'jeden'})
    # slownik.append({'annotation': 'dwa'})
    # slownik.append({'label': 0})
    #
    # slownik.append({'image': '1jeden'})
    # slownik.append({'annotation': '1dwa'})
    # slownik.append({'label': 1})
    #
    # for j in slownik:
    #     print(j['image'])





    # sort_data()
    # train_or_test_limits()
    # train_or_test_others()

    # path = os.path.abspath('main.py')
    # p = os.path.dirname(path)
    # p1 = os.path.dirname(p)
    # print(p)
    # print(p1)
    # data_dir = os.path.join(p1, "dataset")
    # annotations_dir = os.path.join(data_dir, "annotations")
    # print(annotations_dir)
