from keras.layers import Input
from keras.models import Model
from keras.utils.vis_utils import plot_model

import rcnn.frcnn as nn
from rcnn.Configure import Configure

C = Configure()
input_shape = (C.img_height, C.img_width, 3)
anchor_ratios = C.anchor_ratios
anchor_sizes = C.anchor_sizes

cls_count = 4
img_input = Input(input_shape)
roi_input = Input((None, 4))
feature_extractor = nn.back_bone_nn(img_input, trainable=True)
num_anchors = len(anchor_ratios) * len(anchor_sizes)
rpn = nn.rpn(feature_extractor, num_anchors)
classifier = nn.classifier(feature_extractor, roi_input, C.num_rois, nb_classes=cls_count, trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_cls = Model([img_input, roi_input], classifier)
model_all = Model([img_input, roi_input], rpn[:2], classifier)

model_all.summary()
plot_model(model_all, to_file='model_all.jpg', show_shapes=True, show_layer_names=True)
