import cv2
import numpy as np

from display import get_img_annotation

img_width, img_height = (480, 640)
# note that down scale means the ratio along singe single height or width
down_scale = 1/4
# 40,30
anchor_ratios = [1, 0.5, 1.5]
anchor_sizes = [128., 512., 2048.]
iou_label_threshold = 0.5


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
    dw = int(w*dscale)
    dh = int(h*dscale)
    for ibox in boxes:
        # no need for class label
        ibox = np.array(ibox[1:]).astype('float32')
        dboxes.append(ibox*dscale)
    if detail:
        # for detail study
        print(dboxes)
        canv = np.zeros((dw, dh, 3), dtype="uint8")
        for ibox in dboxes:
            cv2.rectangle(canv, (ibox[0], ibox[1]), (ibox[2], ibox[3]), (0, 255, 0))
        # canv = np.zeros((dw, dh, 3), dtype="uint8")
        # show anchors
        for iratio in ratios:
            clr = get_random_clr()
            for isize in anchor_sizes:
                cnt = 0
                ah = int(np.sqrt(iratio*isize))
                aw = int(isize/ah)
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
                        if ix == iy == min(dh, dw)//2:
                            # print((axmin, aymin), (axmax, aymax))
                            cv2.rectangle(canv, (axmin, aymin), (axmax, aymax), clr)
        cv2.imshow('BBox Downscale', canv)
        cv2.waitKey(0)
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


# cal_rpn_y(boxes, w, h, dscale, ratios, sizes)
(img, test_boxex) = get_img_annotation(1, root='../')
cal_rpn_y(test_boxex, img_width, img_height, down_scale, anchor_ratios, anchor_sizes,)






