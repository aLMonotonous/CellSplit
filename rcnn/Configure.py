class Configure:
    def __init__(self):
        self.img_height = 480
        self.img_width = 640
        # note that down scale means the ratio along singe single height or width
        self.down_scale = 1. / 8
        # 30,40
        # h/w
        self.anchor_ratios = [(1, 1), (1, 2), (1.5, 2)]

        self.anchor_sizes = [8., 16., 32.]
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
        # 'whole'/'step'/'FG'
        self.data_load_type = 'whole'
