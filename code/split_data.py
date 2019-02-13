import cv2
import os
import random
import shutil
import numpy as np


def main():
    source_path_root = "data/NKI_Images"
    extract_path_root = "data/NKI_Labeled_Extract"
    split_root = "data/NKI"
    create_dir(split_root)
    train_path_root = os.path.join(split_root, "train")
    create_dir(train_path_root)
    create_dir(os.path.join(train_path_root, "image"))
    create_dir(os.path.join(train_path_root, "label"))
    test_path_root = os.path.join(split_root, "test")
    create_dir(test_path_root)
    create_dir(os.path.join(test_path_root, "image"))
    create_dir(os.path.join(test_path_root, "label"))

    for rt, dirs, files in os.walk(source_path_root):
        for f in files:
            source_path = os.path.join(rt, f)
            extract_path = os.path.join(extract_path_root, f)
            if random.random() > 0.2:
                split = "train"
            else:
                split = "test"

            image = os.path.join(split_root, split, "image", f)
            shutil.copyfile(source_path, image)

            for i in range(10):
                [dirname, filename] = os.path.split(extract_path)
                name = os.path.splitext(filename)[0] + "_c" + str(i) + ".jpg"
                label_src = os.path.join(dirname, name)
                label_target = os.path.join(split_root, split, "label", name)
                shutil.copyfile(label_src, label_target)

    pass


def create_dir(root):
    if not os.path.exists(root):
        os.mkdir(root)


if __name__ == '__main__':
    main()
