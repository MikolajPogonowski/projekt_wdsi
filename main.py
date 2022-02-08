# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import re
import shutil
import glob
import re
from xml.dom import minidom
import xml.etree.ElementTree as ElementTree
#import PIL.Image as Image
import matplotlib.pyplot as plt
import lxml.etree as etree

def copy_data(loc_ori, loc_target, filename, filename_xml):
    original = os.path.join(loc_ori, "images", filename)
    target = os.path.join(loc_target, "images", filename)
    shutil.copyfile(original, target)
    original_xml = os.path.join(loc_ori, "annotations", filename_xml)
    target_xml = os.path.join(loc_target, "annotations", filename_xml)
    shutil.copyfile(original_xml, target_xml)

def import_data():
    data_dir = "dataset"
    annotations_dir = os.path.join(data_dir, "annotations")
    images_dir = os.path.join(data_dir, "images")
    annotations_files = os.listdir(annotations_dir) #ładuje wszystkie nazwy plików z folderu do listy

    #file_name = annotations_files[1]
    #full_file = os.path.join('dataset','annotations', file_name)

    for k in glob.glob('dataset/annotations/*.xml'):
        temp = ElementTree.parse(k)
        #print(temp)
        flag = 0

        obj = temp.findall('filename')
        for j in obj:
            filename = j.text #uzyskujemy nazwę obecnie sprawdzanego pliku png
            filename_xml = re.sub('.png', '.xml', filename) #uzyskujemy nazwę pliku xml
            print(filename)

        obj = temp.findall('object/name')
        for i in obj:
            name = i.text
            print(name)

            if name in ['speedlimit']:
                flag = 1

        if (flag == 1):
            copy_data('dataset', 'limits', filename, filename_xml)
        else:
            copy_data('dataset', 'others', filename, filename_xml)

        #   if (flag == 1):
        #     original = os.path.join("dataset", "images", filename)
        #     target = os.path.join("limits", "images", filename)
        #     shutil.copyfile(original, target)
        #
        #     original_xml = os.path.join("dataset", "annotations", filename_xml)
        #     target_xml = os.path.join("limits", "annotations", filename_xml)
        #     shutil.copyfile(original_xml, target_xml)
        # else:
        #     original = os.path.join("dataset", "images", filename)
        #     target = os.path.join("others", "images", filename)
        #     shutil.copyfile(original, target)
        #
        #     original_xml = os.path.join("dataset", "annotations", filename_xml)
        #     target_xml = os.path.join("others", "annotations", filename_xml)
        #     shutil.copyfile(original_xml, target_xml)

        print(flag)
        print('********************************')



    # original = r'dataset\annotations\road0.xml'
    # target = r'limits\annotations\road0.xml'
    #
    # shutil.copyfile(original, target)


    # sample_annotation = etree.parse(os.path.join(annotations_dir, 'road0.xml'))
    # root = sample_annotation.getroot()
    # print(root)
    # print(root[0].folder)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    import_data()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
