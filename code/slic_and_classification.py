# -*- coding: utf-8 -*-
import gzip
import os
import pickle
from collections import namedtuple

import cv2
import numpy as np
from skimage import data, segmentation, color

from help import class_color

ClassData = namedtuple('ClassData', ['img_name', 'slic_labels', 'cells_labels'])


def main():
    root_dir = r'./data/NKI/test/'

    image_dir = os.path.join(root_dir, "image")
    # label_dir = r'./data/NKI/train/label'
    result_dir = os.path.join(root_dir, "result")
    img_files = [i for i in os.listdir(image_dir) if i.endswith('.jpg')]

    for i, fn in enumerate(img_files):
        print(i)

        img_path = os.path.join(image_dir, fn)

        c_save_dir = [os.path.join(result_dir, "c{0}".format(c_id),
                                   os.path.splitext(fn)[0] + "_c{0}".format(c_id) + ".bmp") for c_id in range(10)]

        result_img = [cv2.imread(r_path, cv2.IMREAD_GRAYSCALE) for r_path in c_save_dir]
        raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        slic_labels = segmentation.slic(raw_img, compactness=30, n_segments=800)
        labels_max = np.max(slic_labels)

        cells_labels = np.zeros_like(slic_labels)

        for labels_index in range(labels_max):

            mask = slic_labels == labels_index
            # mask_rep = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
            # mask_rep = np.repeat(mask_rep, 3, axis=2)
            result_count = []
            for c_id in range(10):
                result = result_img[c_id]
                result = np.bitwise_and(mask, result)
                # cv2.imshow("result",result*255)
                fg_count = np.count_nonzero(result)
                result_count.append(fg_count)
            max_index = result_count.index(max(result_count))
            cells_labels[mask] = max_index

        cd = ClassData(img_name=fn, slic_labels=slic_labels, cells_labels=cells_labels)
        with gzip.GzipFile(os.path.join(root_dir, "classed", "{0}.gz.pkl".format(os.path.splitext(fn)[0])), 'wb') as f:
            pickle.dump(cd, f, True)


def output_img():
    class_color_bgr = np.array(class_color)
    class_color_bgr = class_color_bgr[..., ::-1]
    root_dir = r'./data/NKI/train/'

    image_dir = os.path.join(root_dir, "image")
    # label_dir = r'./data/NKI/train/label'
    result_img_dir = os.path.join(root_dir, "result_final_img")
    img_files = [i for i in os.listdir(image_dir) if i.endswith('.jpg')]

    for i, fn in enumerate(img_files):
        print(i)
        img_path = os.path.join(image_dir, fn)
        classed_path = os.path.join(root_dir, "classed", "{0}.gz.pkl".format(os.path.splitext(fn)[0]))
        with gzip.GzipFile(classed_path, 'rb') as f:
            cd = pickle.load(f)
        label_img = color.label2rgb(cd.cells_labels, colors=class_color_bgr)
        cv2.imwrite(os.path.join(result_img_dir, fn), label_img)


if __name__ == "__main__":
    main()
