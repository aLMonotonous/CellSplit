# from keras.models import load_model
import cv2
import numpy as np
from keras.layers import Input
from keras.models import Model

import rcnn.frcnn as nn
from rcnn.Configure import Configure


def load_rpn_model(C):
    weight_path = C.model_path
    # print(os.listdir(path))
    num_anchors = len(C.anchor_ratios) * len(C.anchor_sizes)
    input_shape = (C.img_height, C.img_width, 3)
    img_input = Input(input_shape)
    feature_extractor = nn.back_bone_nn(img_input, trainable=True)

    rpn = nn.rpn(feature_extractor, num_anchors)
    model_rpn = Model(img_input, rpn[:2])
    model_rpn.load_weights(weight_path)

    return model_rpn


def apply_regr_np(X, T):
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def rpn_decode(cls_target, rgr_target, C, usr_rgr=True):
    anchor_sizes = C.anchor_sizes
    anchor_ratios = C.anchor_ratios
    (rows, cols) = cls_target.shape[1:3]
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
    curr_layer = 0
    for isize in anchor_sizes:
        for iratio in anchor_ratios:
            ah = int(np.sqrt(iratio * isize))
            aw = int(isize / ah)
            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            A[0, :, :, curr_layer] = X - ah / 2
            A[1, :, :, curr_layer] = Y - aw / 2
            A[2, :, :, curr_layer] = ah
            A[3, :, :, curr_layer] = aw
            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)


if __name__ == '__main__':
    C = Configure()
    img_path = '../data/JPEGImages/6.jpg'
    # all_imgs = DG.load_data('../', C)
    # test_imgs = [s for s in all_imgs if s['dataset'] == 'train']
    # data_gen_test = DG.get_rpn_target(test_imgs, C)
    # X, Y = next(data_gen_test)
    img = cv2.imread(img_path)
    img = np.expand_dims(img, axis=0)
    rpn_model = load_rpn_model(C)
    res = rpn_model.predict(img)
    for i in res:
        print(np.shape(i))
