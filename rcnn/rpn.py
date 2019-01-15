import cv2
import numpy as np

from display import get_img_annotation

img_width, img_height = (480, 640)
# note that down scale means the ratio along singe single height or width
down_scale = 1 / 4
# 40,30
anchor_ratios = [1, 0.5, 1.5]
anchor_sizes = [128., 512., 2048.]
iou_label_threshold = 0.5


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


def cal_rpn_y(boxes, w, h, dscale, ratios, sizes, detail=False):
    """
     calculate training target for region proposal network
    :param boxes: ground truth bounding boxs, list-like
    :param w: image width
    :param h: image height
    :param dscale: down sample scale after Feature Extractor
    :param ratios: anchors boxes ratios
    :param sizes:  anchor boxes sizes
    :return:
        target_rgr : bounding box regression parameters for each bbox and best fitting anchor
            feature map size * boxes per point * 4
        target_cls : class-agnostic
            feature map size * boxes per point * 2
    """
    dboxes = []

    dw = int(w * dscale)
    dh = int(h * dscale)

    n_boxes = len(boxes)
    best_iou_bbox = np.zeros(n_boxes).astype('float32')
    best_anchor_bbox = np.zeros([n_boxes, 4]).astype('int')

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


    for iratio in ratios:
        for isize in anchor_sizes:
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
                    cur_anchor = [axmin, aymin, axmax, aymax]
                    # cal IoU for every bbox
                    for idx_dbox, ib in enumerate(dboxes):
                        iou = intersection(cur_anchor, ib) / union(cur_anchor, ib)
                        if iou > best_iou_bbox[idx_dbox]:
                            best_iou_bbox[idx_dbox] = iou
                            best_anchor_bbox[idx_dbox] = cur_anchor
    if detail:
        print(best_iou_bbox, best_anchor_bbox)
        for b in best_anchor_bbox:
            cv2.rectangle(canv, (b[0], b[1]), (b[2], b[3]), get_random_clr())

    if detail:
        cv2.imshow('BBox Downscale', canv)
        cv2.waitKey(0)






# cal_rpn_y(boxes, w, h, dscale, ratios, sizes)
(img, test_boxex) = get_img_annotation(1, root='../')
cal_rpn_y(test_boxex, img_width, img_height, down_scale, anchor_ratios, anchor_sizes,True )
