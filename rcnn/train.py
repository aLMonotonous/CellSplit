from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import generic_utils

import rcnn.frcnn as nn
from rcnn import losses as losses
from rcnn.Configure import Configure

C = Configure()
input_shape = (C.img_height, C.img_width, 3)
anchor_ratios = C.anchor_ratios
anchor_sizes = C.anchor_sizes

cls_count = C.n_cls
img_input = Input(input_shape)
roi_input = Input((None, 4))
# get backbone cnn
feature_extractor = nn.back_bone_nn(img_input, trainable=True)
num_anchors = len(anchor_ratios) * len(anchor_sizes)

rpn = nn.rpn(feature_extractor, num_anchors)
classifier = nn.classifier(feature_extractor, roi_input, C.num_rois, nb_classes=cls_count, trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_cls = Model([img_input, roi_input], classifier)
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# model_all.summary()
# plot_model(model_rpn, to_file='model_rpn.jpg', show_shapes=True, show_layer_names=True)
# plot_model(model_cls, to_file='model_cls.jpg', show_shapes=True, show_layer_names=True)
# plot_model(model_all, to_file='model_all.jpg', show_shapes=True, show_layer_names=True)

# train rpn
model_rpn.load_weights(C.basenet_path, by_name=True)

opt_rpn = Adam(lr=1e-5)
loss_rpn = [losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)]
model_rpn.compile(optimizer=opt_rpn, loss=loss_rpn, metrics={'dense_class_{}'.format(C.n_cls): 'accuracy'})

num_epochs = 16
for idx_epoch in range(num_epochs):
    progbar = generic_utils.Progbar(num_epochs)
    print('Epoch {}/{}'.format(idx_epoch + 1, num_epochs))
    while True:
        try:

