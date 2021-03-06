import random
import time

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

import rcnn.data_generator as DG
import rcnn.frcnn as nn
from rcnn import losses as losses
from rcnn.Configure import Configure

C = Configure()
input_shape = (None, None, 3)
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

model_rpn.summary()
# plot_model(model_rpn, to_file='model_rpn.jpg', show_shapes=True, show_layer_names=True)
# plot_model(model_cls, to_file='model_cls.jpg', show_shapes=True, show_layer_names=True)
# plot_model(model_all, to_file='model_all.jpg', show_shapes=True, show_layer_names=True)

# train rpn
model_rpn.load_weights(C.basenet_path, by_name=True)

opt_rpn = Adam(lr=1e-5)
loss_rpn = [losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)]
model_rpn.compile(optimizer=opt_rpn, loss=loss_rpn, metrics={'dense_class_{}'.format(C.n_cls): 'accuracy'})

num_epochs = 10

all_imgs = DG.load_data('../', C)
random.shuffle(all_imgs)
train_imgs = [s for s in all_imgs if s['dataset'] == 'train']
test_imgs = [s for s in all_imgs if s['dataset'] == 'test']

data_gen_train = DG.get_rpn_target(train_imgs, C)
data_gen_test = DG.get_rpn_target(test_imgs, C)

epoch_length = 20
losses = np.zeros((epoch_length, 5))
print("Start Training")
best_loss = np.Inf
if C.data_load_type == 'whole':
    save_path_x = '../data/merge_data/x.npy'
    save_path_rgr = '../data/merge_data/rgr.npy'
    save_path_cls = '../data/merge_data/cls.npy'
    x_all = np.load(save_path_x)
    cls_all = np.load(save_path_cls)
    rgr_all = np.load(save_path_rgr)
    data_idx = 0
# X, Y = next(data_gen_train)
for idx_epoch in range(num_epochs):
    # progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(idx_epoch + 1, num_epochs))
    start_time = time.time()
    iter_num = 0
    while True:
        try:
            if C.data_load_type == 'FG':
                # test of fit_generator, it still work slowly
                model_rpn.fit_generator(data_gen_train, steps_per_epoch=20, epochs=10)
                break
            elif C.data_load_type == 'whole':
                X = x_all[data_idx]
                # X = np.transpose(X, (0, 2, 1, 3))
                Y = [cls_all[data_idx], rgr_all[data_idx]]
                print('using {}th data to train'.format(data_idx))
                data_idx += 1
            else:
                model_rpn.fit_generator(data_gen_train)
                X, Y = next(data_gen_train)

            # print(X.shape, Y[0].shape, Y[1].shape)
            loss_rpn = model_rpn.train_on_batch(X, Y)
            P_rpn = model_rpn.predict_on_batch(X)

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            # progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])),
            #                           ('rpn_regr', np.mean(losses[:iter_num, 1]))])
            iter_num += 1
            # print(iter_num)
            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                duration = time.time() - start_time
                print(idx_epoch, loss_rpn_cls, loss_rpn_regr, duration)
                curr_loss = loss_rpn_cls + loss_rpn_regr
                if curr_loss < best_loss:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model_rpn.save_weights(C.model_path)
                break



        except Exception as e:
            print("e", e)
            exit()
