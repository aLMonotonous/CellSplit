img_width, img_height = (640, 480)
down_scale = 1/16.
# 40,30
anchor_ratios = [1, 0.5, 1.5]
anchor_sizes = [4, 16]


def cal_rpn_y(boxes, w, h, dscale, ratios, sizes):
    """
     calculate training target for region proposal network
    :param boxes: ground truth bounding boxs, list-like
    :param w: image width
    :param h: image height
    :param dscale: down sample scale after Feature Extractor
    :param ratios: anchors boxes ratios
    :param sizes:  anchor boxes sizes
    :return:
        target_rgr : bounding box regression parameters for each bbox
            feature map size * boxes per point * 2
        target_cls : class-agnostic
            feature map size * boxes per point * 4
    """


# cal_rpn_y(boxes, w, h, dscale, ratios, sizes)






