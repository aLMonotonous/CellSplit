import random
import time

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

import rcnn.data_generator as DG
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
plot_model(model_rpn, to_file='model_rpn.jpg', show_shapes=True, show_layer_names=True)
plot_model(model_cls, to_file='model_cls.jpg', show_shapes=True, show_layer_names=True)
plot_model(model_all, to_file='model_all.jpg', show_shapes=True, show_layer_names=True)

# train rpn
model_rpn.load_weights(C.basenet_path, by_name=True)

opt_rpn = Adam(lr=1e-5)
loss_rpn = [losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)]
model_rpn.compile(optimizer=opt_rpn, loss=loss_rpn, metrics={'dense_class_{}'.format(C.n_cls): 'accuracy'})

num_epochs = 1

all_imgs = DG.load_data('../', C)
random.shuffle(all_imgs)
train_imgs = [s for s in all_imgs if s['dataset'] == 'train']
test_imgs = [s for s in all_imgs if s['dataset'] == 'test']

data_gen_train = DG.get_rpn_target(train_imgs, C)
data_gen_test = DG.get_rpn_target(test_imgs, C)

epoch_length = 10
losses = np.zeros((epoch_length, 5))
print("Start Training")
best_loss = np.Inf
# X, Y = next(data_gen_train)
for idx_epoch in range(num_epochs):
    # progbar = generic_utils.Progbar(num_epochs)
    print('Epoch {}/{}'.format(idx_epoch + 1, num_epochs))
    start_time = time.time()
    iter_num = 0
    while True:
        try:
            X, Y = next(data_gen_train)
            print(X.shape, Y[0].shape, Y[1].shape)
            loss_rpn = model_rpn.train_on_batch(X, Y)
            P_rpn = model_rpn.predict_on_batch(X)

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            # progbar.update(iter_num, ('rpn_cls', np.mean(losses[:iter_num, 0])),
            #                ('rpn_regr', np.mean(losses[:iter_num, 1])))
            iter_num += 1
            print(iter_num)
            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                duration = time.time() - start_time
                print(idx_epoch, loss_rpn_cls, loss_rpn_regr, duration)
                break
                curr_loss = loss_rpn_cls + loss_rpn_regr
                if curr_loss < best_loss:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model_rpn.save_weights(C.model_path)


        except Exception as e:
            print("e", e)
            continue
