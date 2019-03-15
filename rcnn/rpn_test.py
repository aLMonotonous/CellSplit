# from keras.models import load_model
import cv2
import numpy as np
from keras.layers import Input
from keras.models import Model

import rcnn.frcnn as nn
from rcnn.Configure import Configure
from display import show_img_annotation


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


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs


def rpn_decode(cls_target, rgr_target, C, use_rgr=True):
    anchor_sizes = C.anchor_sizes
    anchor_ratios = C.anchor_ratios
    (rows, cols) = cls_target.shape[1:3]
    A = np.zeros((4, cls_target.shape[1], cls_target.shape[2], cls_target.shape[3]))
    curr_layer = 0
    for isize in anchor_sizes:
        for iratio in anchor_ratios:
            ah = int(np.sqrt(iratio * isize))
            aw = int(isize / ah)
            rgr = rgr_target[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
            rgr = np.transpose(rgr, (2, 0, 1))
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            A[0, :, :, curr_layer] = X - ah / 2
            A[1, :, :, curr_layer] = Y - aw / 2
            A[2, :, :, curr_layer] = ah
            A[3, :, :, curr_layer] = aw
            if use_rgr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], rgr)
            # ensure values are reasonable
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])

            all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
            all_probs = cls_target.transpose((0, 3, 1, 2)).reshape((-1))

            x1 = all_boxes[:, 0]
            y1 = all_boxes[:, 1]
            x2 = all_boxes[:, 2]
            y2 = all_boxes[:, 3]

            idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

            all_boxes = np.delete(all_boxes, idxs, 0)
            all_probs = np.delete(all_probs, idxs, 0)

            curr_layer += 1
            result = non_max_suppression_fast(all_boxes, all_probs)[
                0]

            return result


def rpn_decode(cls_target, rgr_target, C):
    cls = cls_target[0, :, :, :]
    rgr = rgr_target[0, :, :, :]
    boxes = []
    for ix in range(cls.shape[0]):
        for iy in range(cls.shape[1]):
            for ia in range(cls.shape[2]):
                if cls[ix, iy, ia] > 0.5:
                    print(ix, iy, ia)
                    box_x = ix
                    box_y = iy
                    ratio = C.anchor_ratios[ia % len(C.anchor_sizes)]
                    size = C.anchor_sizes[int(ia / len(C.anchor_sizes))]
                    h = int(np.sqrt(size * ratio))
                    w = int(size / h)
                    boxes.append([box_x, box_y, h, w])
    return np.array(boxes)


def test():
    C = Configure()
    img_path = '../data/JPEGImages/1.jpg'
    # all_imgs = DG.load_data('../', C)
    # test_imgs = [s for s in all_imgs if s['dataset'] == 'train']
    # data_gen_test = DG.get_rpn_target(test_imgs, C)
    # X, Y = next(data_gen_test)
    img = cv2.imread(img_path)
    img = np.expand_dims(img, axis=0)
    rpn_model = load_rpn_model(C)
    res = rpn_model.predict(img)
    #
    # save_path_rgr = '../data/merge_data/rgr.npy'
    # save_path_cls = '../data/merge_data/cls.npy'
    # cls_all = np.load(save_path_cls)
    # rgr_all = np.load(save_path_rgr)

    # res = [cls_all[0], rgr_all[1]]
    print(res[0].shape, res[1].shape)
    boxes = rpn_decode(res[0], res[1], C)
    print(np.shape(boxes))
    boxes *= 8
    show_img_annotation(img_path, boxes, False)


if __name__ == '__main__':
    test()
