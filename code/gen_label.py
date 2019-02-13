# -*- coding: utf-8 -*-

import os
# os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import numpy as np
import cv2
import mxnet as mx
import matplotlib.pyplot as plt
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])

seg_data_shape = 512

cls_mean_val = np.array([[[125]], [[125]], [[125]]])
cls_std_scale = 1.0

ctx = mx.gpu(0)


def get_segmentation_mod(c_id):
    sym, arg_params, aux_params = mx.model.load_checkpoint("savemodel/nki_c{0}".format(c_id), 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 512, 512))], label_shapes=None)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    return mod


def seg_img(img, mod):
    raw_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(raw_img2, (2, 0, 1))
    img = img[np.newaxis, :]
    img = cls_std_scale * (img.astype(np.float32) - cls_mean_val)

    mod.forward(Batch([mx.nd.array(img)]))
    pred = mod.get_outputs()[0].asnumpy()
    pred = np.argmax(pred, axis=1)[0]

    return pred


def main():
    test_image_dir = r'./data/NKI/test/image'
    test_label_dir = r'./data/NKI/test/label'
    save_dir = r'./data/NKI/test/result'
    img_files = [i for i in os.listdir(test_image_dir) if i.endswith('.jpg')]
    for c_id in range(10):
        c_save_dir = os.path.join(save_dir, "c{0}".format(c_id))
        if not os.path.exists(c_save_dir):
            os.mkdir(c_save_dir)
        seg_mod = get_segmentation_mod(c_id)
        for i, fn in enumerate(img_files):
            fn_path = os.path.join(test_image_dir, fn)
            save_name = os.path.splitext(fn)[0] + "_c{0}".format(c_id) + ".bmp"
            save_path = os.path.join(c_save_dir, save_name)
            raw_img = cv2.imread(fn_path)

            pred = seg_img(raw_img, seg_mod)
            pred = pred * 255
            cv2.imwrite(save_path, pred)


if __name__ == "__main__":
    main()
