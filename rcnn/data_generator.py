import itertools

import cv2
import numpy as np

from display import get_img_annotation
from rcnn.Configure import Configure


def union(au, bu):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - intersection(au, bu)
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def get_random_clr():
    # rgbl = [np.random, 0, 0]
    # np.random.shuffle(rgbl)
    return list(np.random.random(size=3) * 256)


def corner2centre(corner_rprt):
    """
    change corner representation to  centre repr.
    :param corner:array (xmin, ymin, xmax, ymax)
    :return:
    """
    (xmin, ymin, xmax, ymax) = corner_rprt
    cx = xmin + xmax // 2.0
    cy = ymin + ymax // 2.
    height = xmax - xmin
    width = ymax - ymin
    return cx, cy, height, width


def cal_rgr_target(gtbox, anchor):
    acx, acy, ach, acw = corner2centre(anchor)
    boxcx, boxcy, boxh, boxw = corner2centre(gtbox)
    rgr_cx = (boxcx - acx + 0.0) / ach
    rgr_cy = (boxcy - acy + 0.0) / acw
    rgr_h = np.log(boxh + 0.0 / ach)
    rgr_w = np.log(boxw + 0.0 / acw)
    # rgr_h = boxh + 0.0 / ach
    # rgr_w = boxw + 0.0 / acw
    return rgr_cx, rgr_cy, rgr_h, rgr_w


def cal_anchor_index(x, y, idx_sizes, index_ratio, lx, ly, lis, lir):
    # idx =
    pass


def reverse_anchor_shape(index, ratios, sizes):
    """
    reverse shape from index
    :param index: [x,y,ratio+size*len(ratios)]
    :param ratios:
    :param sizes:
    :return:
    """
    cx = index[0]
    cy = index[1]
    ratio = ratios[index[2] % len(ratios)]
    size = sizes[index[2] // len(ratios)]
    h = int(np.sqrt(ratio * size))
    w = int(size / h)
    xmin = cx - h // 2
    xmax = cx + h // 2
    ymin = cy - w // 2
    ymax = cy + 2 // 2
    return xmin, ymin, xmax, ymax


def cal_rpn_y(boxes, C, detail=False):
    """
     calculate training target for region proposal network
     anchor order: x,y,size,ratio
     anchor[x,y,size_num,ratio_num]
    :param boxes: ground truth bounding boxs, list-like
    :return:
        target_rgr : bounding box regression parameters for each bbox and best fitting anchor
            feature map size * boxes per point * 4
        target_cls : class-agnostic
            feature map size * boxes per point * 2
    """
    dboxes = []

    dw = int(C.img_width * C.down_scale)
    dh = int(C.img_height * C.down_scale)
    ol_min = C.ol_range[0]
    ol_max = C.ol_range[1]

    n_boxes = len(boxes)
    best_iou_bbox = np.zeros(n_boxes).astype('float32')
    best_anchor_bbox = np.zeros([n_boxes, 4]).astype('int')
    best_rgr_bbox = np.zeros([n_boxes, 4]).astype('float32')
    # index:[x,y,anchor_index]
    best_anchor_index = np.zeros([n_boxes, 3]).astype('uint8')

    # n_anchors = dw * dh *
    anchor_cls_target = np.zeros([dh, dw, len(C.anchor_ratios) * len(C.anchor_sizes)])
    anchor_rgr_target = np.zeros([dh, dw, 4 * len(C.anchor_ratios) * len(C.anchor_sizes)])

    # record is this anchor valid for cls training
    anchor_valid_cls = np.zeros([dh, dw, len(C.anchor_ratios) * len(C.anchor_sizes)]).astype('uint8')

    # record is this anchor have bigger overlap with bbox
    # that determines  if it is valid for a bbox rgr
    anchor_overlap_rgr = np.zeros([dh, dw, len(C.anchor_ratios) * len(C.anchor_sizes)]).astype('uint8')
    for ibox in boxes:
        # no need for class label
        ibox = np.array(ibox[1:]).astype('float32')
        dboxes.append(ibox * C.down_scale)
    if detail:
        # for detail study
        # print(dboxes)
        canv = np.zeros((dh, dw, 3), dtype="uint8")
        for ibox in dboxes:
            cv2.rectangle(canv, (ibox[0], ibox[1]), (ibox[2], ibox[3]), (0, 255, 0))
        # canv = np.zeros((dw, dh, 3), dtype="uint8")
        # show anchors
        # for iratio in C.anchor_ratios:
        #
        #     for isize in anchor_sizes:
        #         cnt = 0
        #         ah = int(np.sqrt(iratio * isize))
        #         aw = int(isize / ah)
        #         for ix in range(dh):
        #             axmin = int(ix - ah / 2)
        #             axmax = int(ix + ah / 2)
        #             # ignore the box across the boundary of feature map
        #             if axmin < 0 or axmax > dh:
        #                 continue
        #             for iy in range(dw):
        #                 # for every position of Feature map, generate an anchor
        #                 aymin = int(iy - aw / 2)
        #                 aymax = int(iy + aw / 2)
        #                 if aymin < 0 or aymax > dw:
        #                     continue
        #                 cnt += 1
        #                 # cv2.rectangle(canv, (aymin, axmin), (aymax, axmax,), get_random_clr())
        #                 # if cnt < 2:
        #                 # if ix == iy == min(dh, dw) // 2:
        #                 #     cv2.rectangle(canv, (axmin, aymin), (axmax, aymax), get_random_clr())
    for idx_size, isize in enumerate(C.anchor_sizes):
        for idx_ratio, iratio in enumerate(C.anchor_ratios):
            ah = isize * iratio[0]
            aw = isize * iratio[1]
            idx_anchor = idx_ratio + idx_size * len(C.anchor_ratios)
            # print(ah,aw)
            for ix in range(dw):
                axmin = int(ix - aw / 2)
                axmax = int(ix + aw / 2)
                # ignore the box across the boundary of feature map
                if axmin < 0 or axmax > dw:
                    continue
                for iy in range(dh):
                    # for every position of Feature map, generate an anchor
                    aymin = int(iy - ah / 2)
                    aymax = int(iy + ah / 2)
                    if aymin < 0 or aymax > dh:
                        continue
                    # anchor_index = cal_anchor_index()
                    cur_anchor = [axmin, aymin, axmax, aymax]
                    cur_anchor_idx = [iy, ix, idx_anchor]
                    # cal IoU for every bbox
                    for idx_dbox, ib in enumerate(dboxes):
                        iou = intersection(cur_anchor, ib) / union(cur_anchor, ib)
                        if iou > ol_max:
                            # this is a valid positive anchor
                            # note that there could be bbox that has no anchor
                            # anchor_overlap_rgr[iy, ix, idx_anchor] = 1
                            # for overlap > threshold , set overlap
                            anchor_cls_target[iy, ix, idx_anchor] = 1
                            anchor_valid_cls[iy, ix, idx_anchor] = 1
                        if iou > best_iou_bbox[idx_dbox]:
                            best_iou_bbox[idx_dbox] = iou
                            # record index for best anchor, could be use for update rgr target
                            best_anchor_bbox[idx_dbox] = cur_anchor
                            best_anchor_index[idx_dbox] = cur_anchor_idx
                            best_rgr_bbox[idx_dbox] = (cal_rgr_target(ib, cur_anchor))
                            # print(best_rgr_bbox)
                            # do not update rgr target here
                            # only best anchor for bbox deserve a rgr para.
                        if iou < ol_min:
                            # this is a valid negative anchor
                            anchor_valid_cls[iy, ix, idx_anchor] = 1
        # after find best anchor for every bbox (suppose),
        # TODO there should be a way to solve when can not find a best anchor for a bbox
        # update rgr target
        for idx_dbox, ib in enumerate(dboxes):
            y, x, idx = best_anchor_index[idx_dbox]
            start = int(idx * 4)
            anchor_overlap_rgr[y, x, idx] = 1
            anchor_rgr_target[y, x, start:start + 4] = best_rgr_bbox[idx_dbox]

    if detail:
        # canv = np.zeros((dh, dw, 3), dtype="uint8")
        print("best iou box")
        print(best_iou_bbox)
        print("best anchor box")
        print(best_anchor_bbox)
        print(dboxes)
        print("best rgr target")
        # print(anchor_rgr_target)
        shape = (dh, dw, len(C.anchor_ratios) * len(C.anchor_sizes))
        for idx in itertools.product(*[range(s) for s in shape]):
            if anchor_rgr_target[idx] != 0:
                print(idx, anchor_rgr_target[idx])
        # print("cls target")
        # print(anchor_cls_target)
        # # show best anchors
        for b in best_anchor_bbox:
            cv2.rectangle(canv, (b[0], b[1]), (b[2], b[3]), get_random_clr())
        shape = (dh, dw, len(C.anchor_ratios) * len(C.anchor_sizes))
        # show valid postive anchors
        # for idx in itertools.product(*[range(s) for s in shape]):
        #     if anchor_cls_target[idx] == 1.0:
        #         b = reverse_anchor_shape(idx, C.anchor_ratios, C.anchor_sizes)
        #         cv2.rectangle(canv, (b[0], b[1]), (b[2], b[3]), get_random_clr())

    if detail:
        cv2.imshow('BBox Downscale', canv)
        cv2.waitKey(0)

    # TODO
    #  Add filters to reduce data imbalance

    anchor_valid_cls = np.transpose(anchor_valid_cls, [2, 0, 1])
    anchor_valid_cls = np.expand_dims(anchor_valid_cls, axis=0)

    anchor_overlap_rgr = np.transpose(anchor_overlap_rgr, [2, 0, 1])
    anchor_overlap_rgr = np.expand_dims(anchor_overlap_rgr, axis=0)

    anchor_rgr_target = np.transpose(anchor_rgr_target, [2, 0, 1])
    anchor_rgr_target = np.expand_dims(anchor_rgr_target, axis=0)
    
    anchor_cls_target = np.transpose(anchor_cls_target, [2, 0, 1])
    anchor_cls_target = np.expand_dims(anchor_cls_target, axis=0)

    pos_locs = np.where(np.logical_and(anchor_overlap_rgr[0, :, :, :] == 1, anchor_valid_cls[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(anchor_overlap_rgr[0, :, :, :] == 0, anchor_valid_cls[0, :, :, :] == 1))

    rgr_y = np.concatenate([np.repeat(anchor_overlap_rgr, 4, axis=1), anchor_rgr_target], axis=1)
    cls_y = np.concatenate([anchor_valid_cls, anchor_cls_target], axis=1)
    # rgr_y = anchor_rgr_target
    #   # cls_y = anchor_cls_target
    # shape :(1,anchors_idx,w,h)
    return cls_y, rgr_y


def get_rpn_target(all_imgs, C, mode='train', detail=False):
    while True:
        if mode == 'train':
            np.random.shuffle(all_imgs)
        for img in all_imgs:
            if img['dataset'] == 'test':
                continue
            img_path = img['filepath']
            bboxes = img['bboxes']

            x_img = cv2.imread(img_path)
            rows, cols, _ = x_img.shape
            y_rpn_cls, y_rpn_rgr = cal_rpn_y(bboxes, C, detail)
            # pre-procession
            # BGR -> RGB
            x_img = x_img[:, :, (2, 0, 1)]
            x_img = x_img.astype(np.float32)
            x_img[:, :, 0] -= C.img_channel_mean[0]
            x_img[:, :, 1] -= C.img_channel_mean[1]
            x_img[:, :, 2] -= C.img_channel_mean[2]
            x_img /= C.img_scaling_factor

            x_img = np.transpose(x_img, (2, 0, 1))
            x_img = np.expand_dims(x_img, axis=0)
            # tf
            x_img = np.transpose(x_img, (0, 2, 3, 1))
            y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
            y_rpn_rgr = np.transpose(y_rpn_rgr, (0, 2, 3, 1))
            # print(y_rpn_cls.shape, y_rpn_rgr.shape )
            y_rpn_cls = y_rpn_cls.astype('float32')
            y_rpn_rgr = y_rpn_rgr.astype('float32')
            yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_rgr)]


def load_data(path, C):
    all_data = []
    all_id = np.arange(C.total_data_num)
    np.random.shuffle(all_id)
    train_test_ratio = 0.66
    mid = int(C.total_data_num * train_test_ratio)
    train_set = all_id[:mid]
    test_set = all_id[mid:]
    for i in range(C.total_data_num):
        (img, test_boxes) = get_img_annotation(i, root=path)
        element = {'filepath': img, 'bboxes': test_boxes, 'dataset': 'train'}
        if i in test_set:
            element['dataset'] = 'test'
        all_data.append(element)
    return all_data

    # cal_rpn_y(boxes, w, h, dscale, ratios, sizes)


if __name__ == '__main__':
    detail = True
    C = Configure()
    load_data('../', C)
    img_width = C.img_width
    img_height = C.img_height
    # note that down scale means the ratio along singe single height or width
    down_scale = C.down_scale
    # 30,40
    anchor_ratios = C.anchor_ratios
    anchor_sizes = C.anchor_sizes
    overlap_max = C.overlap_max
    overlap_min = C.overlap_min
    ol_range = [overlap_min, overlap_max]
    save = True
    if save:
        data_xs = []
        data_cls_ys = []
        data_rgr_ys = []
        save_path_x = '../data/merge_data/x.npy'
        save_path_rgr = '../data/merge_data/rgr.npy'
        save_path_cls = '../data/merge_data/cls.npy'
        all_imgs = load_data('../', C)
        gen = get_rpn_target(all_imgs, C, mode='test', detail=False)
        for i in range(len(all_imgs)):
            print('fixing {}th data'.format(i))
            X, Y = next(gen)
            data_xs.append(X)
            data_cls_ys.append(Y[0])
            data_rgr_ys.append(Y[1])

            lim = 20
            if detail:
                cls = Y[0]
                rgr = Y[1]
                for i in range(lim):
                    for j in range(lim):
                        # if 1 in cls[0, i, j, :]:
                        print(cls[0, i, j, :9])
                        print(cls[0, i, j, 9:])
                        print("-c-" * 8)

                for i in range(lim):
                    for j in range(lim):
                        if 1 in rgr[0, i, j, :]:
                            print(rgr[0, i, j, :36])
                            print(rgr[0, i, j, 36:])
                            print("-r-" * 8)

                print()
                print('cls', Y[0].shape, Y[0][0, 5, 5, :])
                print('rgr', Y[1].shape, Y[1][0, 25, 15, :])
        np.save(save_path_x, data_xs)
        np.save(save_path_rgr, data_rgr_ys)
        np.save(save_path_cls, data_cls_ys)
