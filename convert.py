import os
import random
import xml.etree.ElementTree as ET

import pandas as pd

# import numpy as np


classes = ["RBC", "WBC", "Platelets"]
ratio = 0.66
path = 'data'


def convert_xml(dataset_dir, ratio, classes):
    assert 0 < ratio < 1
    img_dir = os.path.join(dataset_dir, "JPEGImages")
    label_dir = os.path.join(dataset_dir, "Annotations")
    img_names = os.listdir(img_dir)
    img_names.sort()
    label_names = os.listdir(label_dir)
    label_names.sort()
    file_num = len(img_names)
    assert file_num == len(label_names)

    # select train/test
    idx_random = list(range(file_num))
    random.shuffle(idx_random)
    idx_train = idx_random[:int(file_num*ratio)+1]
    idx_val = idx_random[int(file_num*ratio)+1:]

    #writing
    train_df = []
    val_df = []
    list_dir = os.path.join(dataset_dir, "list")
    train_list_dir = os.path.join(list_dir, "train.csv")
    val_list_dir = os.path.join(list_dir, "val.csv")
    # with open(train_list_dir, "w") as train_list:
    #     print("Writing in train.lst")
    #     with open(val_list_dir,"w") as val_list:
    #         print("Writing val.lst")
    for idx in range(file_num):
        # print("Writing No."+str(idx))
        each_image_path = os.path.join(img_dir, img_names[idx])
        each_label_path = os.path.join(label_dir, label_names[idx])
        tree = ET.parse(each_label_path)
        rt = tree.getroot()
        size = rt.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        for obj in rt.iter('object'):
            row = []
            # row.append(str(idx))
            row.append(str(each_image_path))
            row.append(str(width))
            row.append(str(height))
            cls_name = obj.find('name').text
            if cls_name not in classes:
                continue
            # cls_id = classes.index(cls_name)
            cls_id = cls_name
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text) / width
            ymin = float(xml_box.find('ymin').text) / height
            xmax = float(xml_box.find('xmax').text) / width
            ymax = float(xml_box.find('ymax').text) / height
            for i in [cls_id, xmin, ymin, xmax, ymax]:
                row.append(str(i))
            if idx in idx_train:
                train_df.append(row)
            else:
                val_df.append(row)
    d_columns = ['file_path', 'width', 'height', 'class', 'xmin',
                 'ymin', 'xmax', 'ymax']
    train_df = pd.DataFrame(train_df, columns=d_columns)
    train_df.to_csv(train_list_dir)
    val_df = pd.DataFrame(val_df, columns=d_columns)
    val_df.to_csv(val_list_dir)


def read_lst(dataset_dir):
    list_dir = os.path.join(dataset_dir, "list")
    train_list_dir = os.path.join(list_dir, "train.csv")
    train = pd.read_csv(train_list_dir)
    print(train.head())
    # sum
    print(train['file_path'].nunique(),"images in training set")
    print(train['class'].value_counts())

convert_xml(path, ratio, classes)
read_lst(path)


