
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
        self.n_cls = 4
        self.num_rois = 4
        self.total_data_num = 364
        self.basenet_path = '../models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        self.model_path = '../models/rpn.h5'

