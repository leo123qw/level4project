import os

import numpy as np
import cv2

from help import class_color


def main():
    class_center_point = np.array(class_color)
    class_center_point = class_center_point[..., ::-1]

    labeled_path = "data/NKI_Labeled"
    extract_path_root = "data/NKI_Labeled_Extract" # gonna create a new file that include 9 different labels for the same pic.
    if not os.path.exists(extract_path_root):
        os.mkdir(extract_path_root) # if the file is not exist then create a new file otherwise add the image into the file.
    for rt, dirs, files in os.walk(labeled_path):
        for f in files:
            input_path = os.path.join(rt, f)
            extract_path = os.path.join(extract_path_root, f)
            extract_label(class_center_point, input_path, extract_path)
    # cv2.imshow(str(("spaceship K=", k)), res2)
    # cv2.imshow('quondam image', img)
    # cv2.waitKey(0)
#------------------- 根据help.py 文件里的颜色去把数据中染色的颜色去除杂质， 然后提纯， 清晰化， 把染的颜色变成纯色之后好让机器识别---------
#------------------- 产生一个新的文件夹nki_labeled_extract， 文件夹里面保存的是把9种不同颜色的细胞单独提取出来然后变成单独的图片。-------

def extract_label(class_center_point, input_path, extract_path):
    img = cv2.imread(input_path)
    z = img.reshape((-1, 3))
    # convert to np.float32
    z = np.float32(z)
    # define criteria, number of clusters(K) and apply kmeans()
    #My criteria is such that, whenever 10 iterations of algorithm is ran, 
    #or an accuracy of epsilon = 1.0 is reached, stop the algorithm and return the answer.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    k = 10
    ret, label, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image    因为help.py  里就是uint8 所以这里要改回去。
    for i in range(len(centers)):
        center = centers[i]
        distances = get_distances(class_center_point, center)
        argsort = distances.argsort(axis=0)
        centers[i] = class_center_point[argsort[0]]
    centers = np.uint8(centers)
    res = centers[label.flatten()]
    res2 = res.reshape(img.shape)
    extract_path = extract_path.replace(".v2", "")
    cv2.imwrite(extract_path, res2)

    for i, c in enumerate(class_center_point):
        label_img = res2.copy()
        red, green, blue = label_img[:, :, 0], label_img[:, :, 1], label_img[:, :, 2]
        mask = (red == c[0]) & (green == c[1]) & (blue == c[2])
        label_img[:, :, :3][mask] = [255, 255, 255]
        label_img[:, :, :3][~mask] = [0, 0, 0]
        name = os.path.splitext(os.path.basename(extract_path))[0] + "_c" + str(i) + ".jpg"
        [dirname, filename] = os.path.split(extract_path)
        label = os.path.join(dirname, name)
        cv2.imwrite(label, label_img)


def get_distances(point_x, point_t):  # 计算训练集每个点与计算点的欧几米得距离  # 求中心种子点的公式
    points = np.zeros_like(point_t)  # 获得与训练集X一样结构的0集
    points[:] = point_t
    minus_square = (point_x - point_t) ** 2
    euclidean_distances = np.sqrt(minus_square.sum(axis=1))  # 训练集每个点与特殊点的欧几米得距离
    return euclidean_distances


if __name__ == '__main__':
    main()
