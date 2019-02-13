# -*- coding: utf-8 -*-
import gzip
import os
import pickle
import re
from collections import namedtuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.special import comb
from skimage import data, segmentation, color
from skimage.feature import greycomatrix, greycoprops

from help import class_color

ClassData = namedtuple('ClassData', ['img_name', 'slic_labels', 'cells_labels'])


def feature_glcm(raw_img, cells_labels):
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    class_num = 9
    features = []
    for c_id in range(class_num):
        mask = cells_labels == c_id

        mask = (mask * 255).astype(np.uint8)
        bg = cv2.bitwise_not(mask)
        mask_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)
        mask_img = mask_img + bg

        # cv2.imshow('drawimg', mask_img)
        glcm = greycomatrix(mask_img, [5], [0], 256, symmetric=True, normed=True)
        features.append(greycoprops(glcm, 'contrast')[0, 0])
        features.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        features.append(greycoprops(glcm, 'homogeneity')[0, 0])
        features.append(greycoprops(glcm, 'ASM')[0, 0])
        features.append(greycoprops(glcm, 'energy')[0, 0])
        features.append(greycoprops(glcm, 'correlation')[0, 0])

    return features


def feature_distance(distance, labels):
    """
    提取特征
    :param distance:
    :param labels:
    :return:
    """
    class_num = 9
    feature_num = 4
    features = []
    for c_id_x in range(class_num):
        x_mask = labels == c_id_x
        if np.count_nonzero(x_mask) == 0:
            for c_id_y in range(class_num):
                if c_id_x == c_id_y:
                    continue
                for i in range(feature_num):
                    features.append(0)
            continue
        x_d = distance[x_mask]
        for c_id_y in range(class_num):
            if c_id_x == c_id_y:
                continue
            y_mask = labels == c_id_y
            if np.count_nonzero(y_mask) == 0:
                for i in range(feature_num):
                    features.append(0)
                continue

            x_y_d = x_d[:, y_mask]
            # 每个每种细胞与其他细胞的距离的均值和标准差
            features.append(np.mean(x_y_d))
            features.append(np.std(x_y_d))
            # 每个每种细胞与最近的其他细胞的距离的均值和标准差
            min_d = np.min(x_y_d, axis=1)
            features.append(np.mean(min_d))
            features.append(np.std(min_d))

    return features


def main():
    class_color_bgr = np.array(class_color)
    class_color_bgr = class_color_bgr[..., ::-1]

    root_dir = r'./data/NKI/test/'

    image_dir = os.path.join(root_dir, "image")
    # label_dir = r'./data/NKI/train/label'
    feature_dir = os.path.join(root_dir, "feature")
    img_files = [i for i in os.listdir(image_dir) if i.endswith('.jpg')]
    features_dict = {}

    for i, fn in enumerate(img_files):
        print(i)
        features = []
        img_path = os.path.join(image_dir, fn)
        classed_path = os.path.join(root_dir, "classed", "{0}.gz.pkl".format(os.path.splitext(fn)[0]))
        with gzip.GzipFile(classed_path, 'rb') as f:
            cd = pickle.load(f)
        labels_max = np.max(cd.slic_labels)
        points = np.zeros((labels_max, 2))
        labels = np.zeros(labels_max)
        for labels_index in range(labels_max):
            mask = cd.slic_labels == labels_index
            mask = mask * 255
            mask = mask.astype(np.uint8)
            # cv2.imshow('drawimg', mask)
            # image, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            m = cv2.moments(mask, 1)
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            # img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            # l = cd.cells_labels[cy, cx]
            # t = class_color_bgr[l]
            # img = cv2.circle(img, (cx, cy), 6, (int(t[0]), int(t[1]), int(t[2])),3)
            # cv2.imshow('drawimg', img)
            points[labels_index, 0] = cx
            points[labels_index, 1] = cy
            labels[labels_index] = cd.cells_labels[cy, cx]

        points_d = squareform(pdist(points, 'euclidean'))
        f = feature_distance(points_d, labels)
        features.extend(f)

        raw_img = cv2.imread(img_path)
        f = feature_glcm(raw_img, cd.cells_labels)
        features.extend(f)
        fn_r = re.match("([0-9]*?)___*", fn)
        fn_id = fn_r.group(1)
        if fn_id not in features_dict:
            features_dict[fn_id] = []
        features_dict[fn_id].append(features)

    nki_survival = pd.read_csv('./data/NKI/nki_survival.csv', index_col=0)
    nki_data = nki_survival.loc[:, ['Survival_2005']]
    nki_data['Survival_2005'] = nki_data['Survival_2005'] >= 5
    nki_data.index.name = 'rosid'

    for fn_id in features_dict:
        features_dict[fn_id] = np.mean(features_dict[fn_id], axis=0)
        pass

    tmp_v = next(iter(features_dict))
    features_name_list = ["f_{0:0>4}".format(i) for i in range(len(features_dict[tmp_v]))]
    features_pd = pd.DataFrame.from_dict(features_dict, orient='index', columns=features_name_list)
    features_pd.index.name = 'rosid'
    features_pd.index = features_pd.index.astype(np.int64)

    concat = pd.concat([nki_data, features_pd], axis=1)
    concat.to_csv('./data/NKI/test_features.csv')


if __name__ == "__main__":
    main()
