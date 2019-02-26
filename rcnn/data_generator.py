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


def cal_rpn_y(boxes, w, h, dscale, ratios, sizes, overlap_range, detail=False):
    """
     calculate training target for region proposal network
     anchor order: x,y,size,ratio
     anchor[x,y,size_num,ratio_num]
    :param boxes: ground truth bounding boxs, list-like
    :param w: image width
    :param h: image height
    :param dscale: down sample scale after Feature Extractor
    :param ratios: anchors boxes ratios
    :param sizes:  anchor boxes sizes
    :param overlap_range: [min,max]
    :param detail :show detail info
    :return:
        target_rgr : bounding box regression parameters for each bbox and best fitting anchor
            feature map size * boxes per point * 4
        target_cls : class-agnostic
            feature map size * boxes per point * 2
    """
    dboxes = []

    dw = int(w * dscale)
    dh = int(h * dscale)
    ol_min = overlap_range[0]
    ol_max = overlap_range[1]

    n_boxes = len(boxes)
    best_iou_bbox = np.zeros(n_boxes).astype('float32')
    best_anchor_bbox = np.zeros([n_boxes, 4]).astype('int')
    best_rgr_bbox = np.zeros([n_boxes, 4]).astype('float32')
    # index:[x,y,anchor_index]
    best_anchor_index = np.zeros([n_boxes, 3]).astype('uint8')

    # n_anchors = dw * dh *
    anchor_cls_target = -1 * np.ones([dh, dw, len(ratios) * len(sizes)])
    anchor_rgr_target = -1 * np.ones([dh, dw, 4 * len(ratios) * len(sizes)])

    # record is this anchor valid for cls training
    anchor_valid_cls = np.zeros([dh, dw, len(ratios) * len(sizes)]).astype('uint8')

    # record is this anchor have bigger overlap with bbox
    # that determines  if it is valid for a bbox rgr
    anchor_overlap_rgr = np.zeros([dh, dw, len(ratios) * len(sizes)]).astype('uint8')

    for ibox in boxes:
        # no need for class label
        ibox = np.array(ibox[1:]).astype('float32')
        dboxes.append(ibox * dscale)
    if detail:
        # for detail study
        # print(dboxes)
        canv = np.zeros((dw, dh, 3), dtype="uint8")
        for ibox in dboxes:
            cv2.rectangle(canv, (ibox[0], ibox[1]), (ibox[2], ibox[3]), (0, 255, 0))
        # canv = np.zeros((dw, dh, 3), dtype="uint8")
        # show anchors
        for iratio in ratios:
            clr = get_random_clr()
            for isize in anchor_sizes:
                cnt = 0
                ah = int(np.sqrt(iratio * isize))
                aw = int(isize / ah)
                for ix in range(dh):
                    axmin = int(ix - ah / 2)
                    axmax = int(ix + ah / 2)
                    # ignore the box across the boundary of feature map
                    if axmin < 0 or axmax > h:
                        continue
                    for iy in range(dw):
                        # for every position of Feature map, generate an anchor
                        aymin = int(iy - aw / 2)
                        aymax = int(iy + aw / 2)
                        if aymin < 0 or aymax > w:
                            continue
                        cnt += 1
                        # if cnt < 2:
                        if ix == iy == min(dh, dw) // 2:
                            # print((axmin, aymin), (axmax, aymax))
                            pass
                            # cv2.rectangle(canv, (axmin, aymin), (axmax, aymax), clr)
    for idx_size, isize in enumerate(sizes):
        for idx_ratio, iratio in enumerate(ratios):
            ah = int(np.sqrt(iratio * isize))
            aw = int(isize / ah)
            idx_anchor = idx_ratio + idx_size * len(ratios)
            for ix in range(dh):
                axmin = int(ix - ah / 2)
                axmax = int(ix + ah / 2)
                # ignore the box across the boundary of feature map
                if axmin < 0 or axmax > h:
                    continue
                for iy in range(dw):
                    # for every position of Feature map, generate an anchor
                    aymin = int(iy - aw / 2)
                    aymax = int(iy + aw / 2)
                    if aymin < 0 or aymax > w:
                        continue
                    # anchor_index = cal_anchor_index()
                    cur_anchor = [axmin, aymin, axmax, aymax]
                    cur_anchor_idx = [ix, iy, idx_anchor]
                    # cal IoU for every bbox
                    for idx_dbox, ib in enumerate(dboxes):
                        iou = intersection(cur_anchor, ib) / union(cur_anchor, ib)
                        if iou > ol_max:
                            # this is a valid positive anchor
                            # note that there could be bbox that has no anchor
                            anchor_overlap_rgr[ix, iy, idx_anchor] = 1
                            # for overlap > threshold , set overlap
                            anchor_cls_target[ix, iy, idx_anchor] = 1
                            anchor_valid_cls[ix, iy, idx_anchor] = 1
                            if iou > best_iou_bbox[idx_dbox]:
                                best_iou_bbox[idx_dbox] = iou
                                # record index for best anchor, could be use for update rgr target
                                best_anchor_bbox[idx_dbox] = cur_anchor
                                best_anchor_index[idx_dbox] = cur_anchor_idx
                                best_rgr_bbox[idx_dbox] = (cal_rgr_target(ib, cur_anchor))
                                # do not update rgr target here
                                # only best anchor for bbox deserve a rgr para.
                        if iou < ol_min:
                            # this is a valid negative anchor
                            anchor_valid_cls[ix, iy, idx_anchor] = 1
        # after find best anchor for every bbox (suppose),
        # update rgr target
        for idx_dbox, ib in enumerate(dboxes):
            x, y, idx = best_anchor_index[idx_dbox]
            start = int(idx * 4)
            anchor_valid_cls[x, y, start:start + 4] = 1
            anchor_rgr_target[x, y, start:start + 4] = best_rgr_bbox[idx_dbox]

    if detail:
        print("best iou box")
        print(best_iou_bbox)
        print("best anchor box")
        print(best_anchor_bbox)
        print(dboxes)
        print("best rgr target")
        # print(anchor_rgr_target)
        shape = (dh, dw, len(ratios) * len(sizes))
        for idx in itertools.product(*[range(s) for s in shape]):
            if anchor_rgr_target[idx] != -1:
                print(idx, anchor_rgr_target[idx])
        print("cls target")
        print(anchor_cls_target)
        for b in best_anchor_bbox:
            cv2.rectangle(canv, (b[0], b[1]), (b[2], b[3]), get_random_clr())
        shape = (dh, dw, len(ratios) * len(sizes))
        for idx in itertools.product(*[range(s) for s in shape]):
            if anchor_cls_target[idx] == 1.0:
                b = reverse_anchor_shape(idx, ratios, sizes)
                cv2.rectangle(canv, (b[0], b[1]), (b[2], b[3]), get_random_clr())

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

    pos_locs = np.where(np.logical_and(anchor_overlap_rgr[0, :, :, :] == 1, anchor_valid_cls[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(anchor_overlap_rgr[0, :, :, :] == 0, anchor_valid_cls[0, :, :, :] == 1))

    rgr_y = np.concatenate([np.repeat(anchor_overlap_rgr, 4, axis=1), anchor_rgr_target], axis=1)
    cls_y = np.concatenate([anchor_valid_cls, anchor_overlap_rgr], axis=1)
    return rgr_y, cls_y


# cal_rpn_y(boxes, w, h, dscale, ratios, sizes)
if __name__ == '__main__':
    C = Configure()
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
    data_xs = []
    data_cls_ys = []
    data_rgr_ys = []
    save_path_x = 'data\\merge_data\\x.npy'
    save_path_rgr = 'data\\merge_data\\rgr.npy'
    save_path_cls = 'data\\merge_data\\cls.npy'

    for i in range(0, 363):
        print(i)
        (img, test_boxex) = get_img_annotation(i, root='../')
        rgr_y, cls_y = cal_rpn_y(test_boxex, img_width, img_height, down_scale, anchor_ratios, anchor_sizes, ol_range)
        data_xs.append(img)
        data_rgr_ys.append(rgr_y)
        data_cls_ys.append(cls_y)
    np.save(save_path_x, data_xs)
    np.save(save_path_rgr, data_rgr_ys)
    np.save(save_path_cls, data_cls_ys)
