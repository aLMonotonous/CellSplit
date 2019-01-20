
class Configure:
    def __init__(self):
        self.img_width = 640
        self.img_height = 480
        # note that down scale means the ratio along singe single height or width
        self.down_scale = 1 / 8
        # 30,40
        self.anchor_ratios = [1, 0.5, 1.5]
        self.anchor_sizes = [128., 512., 2048.]
        self.overlap_max = 0.7
        self.overlap_min = 0.3
        self.ol_range = [self.overlap_min, self.overlap_max]

        self.num_rois = 4
