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
