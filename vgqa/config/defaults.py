from yacs.config import CfgNode as Cfg


def _build_input_cfg() -> Cfg:
    cfg = Cfg()
    cfg.MAX_QUERY_LEN = 26
    cfg.MAX_VIDEO_LEN = 200
    # frame count per sample
    cfg.TRAIN_SAMPLE_NUM = 64
    # input spatial size
    cfg.RESOLUTION = 224
    # normalization
    cfg.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.PIXEL_STD = [0.229, 0.224, 0.225]
    # augmentations
    cfg.AUG_SCALE = True
    cfg.AUG_TRANSLATE = False
    cfg.FLIP_PROB_TRAIN = 0.5
    cfg.TEMP_CROP_PROB = 0.5
    return cfg


def _build_model_cfg() -> Cfg:
    m = Cfg()
    m.DEVICE = "cuda"
    m.WEIGHT = ""
    m.WEIGHT_EVAL = ""  # separate weight for evaluation
    m.EMA = True
    m.EMA_DECAY = 0.9998
    m.QUERY_NUM = 1  # each frame a single query
    m.DOWN_RATIO = 4

    # vision backbone
    m.VISION_BACKBONE = Cfg()
    m.VISION_BACKBONE.NAME = 'resnet101'  # resnet50 or resnet101
    m.VISION_BACKBONE.POS_ENC = 'sine'  # sine, sineHW or learned
    m.VISION_BACKBONE.DILATION = False  # replace stride with dilation in last block (DC5)
    m.VISION_BACKBONE.FREEZE = False

    # video swin
    m.VIDEO_SWIN = Cfg()
    m.VIDEO_SWIN.MODEL_NAME = 'video_swin_t_p4w7'
    m.VIDEO_SWIN.PRETRAINED = 'video_swin_t_p4w7_k400_1k'
    m.VIDEO_SWIN.FEATURE_DIM = 768
    m.VIDEO_SWIN.FREEZE = True

    # text model
    m.TEXT_MODEL = Cfg()
    m.TEXT_MODEL.NAME = 'roberta-base'  # "bert-base", "roberta-large"
    m.TEXT_MODEL.FREEZE = False

    # optional LSTM encoder
    m.USE_LSTM = False
    m.LSTM = Cfg()
    m.LSTM.NAME = 'lstm'
    m.LSTM.HIDDEN_SIZE = 512
    m.LSTM.BIDIRECTIONAL = True
    m.LSTM.DROPOUT = 0
    m.LSTM_NUM_LAYERS = 2

    # VSTG pipeline
    m.VSTG = Cfg()
    m.VSTG.HIDDEN = 256
    m.VSTG.QUERY_DIM = 4  # anchor dim
    m.VSTG.ENC_LAYERS = 6
    m.VSTG.DEC_LAYERS = 6
    m.VSTG.FFN_DIM = 2048
    m.VSTG.DROPOUT = 0.1
    m.VSTG.HEADS = 8
    m.VSTG.USE_LEARN_TIME_EMBED = False
    m.VSTG.USE_ACTION = True  # actioness head
    m.VSTG.FROM_SCRATCH = True

    # 2D-Map prediction
    m.VSTG.TEMP_PRED_LAYERS = 6
    m.VSTG.CONV_LAYERS = 4
    m.VSTG.TEMP_HEAD = 'attn'  # attn or conv
    m.VSTG.KERNAL_SIZE = 9
    m.VSTG.MAX_MAP_SIZE = 128
    m.VSTG.POOLING_COUNTS = [15, 8, 8, 8]

    return m


def _build_dataset_cfg() -> Cfg:
    d = Cfg()
    d.NAME = 'VidSTG'
    d.NUM_CLIP_FRAMES = 32
    d.MIN_GT_FRAME = 4  # minimum gt frames in a sampled clip
    d.APP_NUM = 20
    d.MOT_NUM = 34
    return d


def _build_dataloader_cfg() -> Cfg:
    dl = Cfg()
    dl.NUM_WORKERS = 4
    dl.SIZE_DIVISIBILITY = 0
    dl.ASPECT_RATIO_GROUPING = False
    return dl


def _build_solver_cfg() -> Cfg:
    s = Cfg()
    s.MAX_EPOCH = 30
    s.BATCH_SIZE = 1  # videos per GPU; should be 1
    s.SHUFFLE = True
    s.BASE_LR = 2e-5
    s.VIS_BACKBONE_LR = 1e-5
    s.TEXT_LR = 2e-5
    s.TEMP_LR = 1e-4
    s.VERB_LR = 3e-3
    s.OPTIMIZER = 'adamw'
    s.MAX_GRAD_NORM = 0.1

    # loss weights
    s.BBOX_COEF = 5
    s.GIOU_COEF = 2
    s.TEMP_COEF = 2
    s.ATTN_COEF = 1
    s.ACTIONESS_COEF = 2
    s.CONF_COEF = 1
    s.CONF2_COEF = 1
    s.CONF3_COEF = 1
    s.CONF4_COEF = 1

    # lr scheduling
    s.MOMENTUM = 0.9
    s.WEIGHT_DECAY = 0.0001
    s.GAMMA = 0.1
    s.POWER = 0.9  # Poly LRScheduler
    s.STEPS = (30000,)
    s.WARMUP_FACTOR = 1.0 / 3
    s.WARMUP_ITERS = 500
    s.WARMUP_PROP = 0.01
    s.WARMUP_METHOD = "linear"

    s.SCHEDULE = Cfg()
    s.SCHEDULE.TYPE = "linear_with_warmup"
    s.SCHEDULE.DROP_STEP = [8, 12]
    # WarmupReduceLROnPlateau-only params
    s.SCHEDULE.PATIENCE = 2
    s.SCHEDULE.THRESHOLD = 1e-4
    s.SCHEDULE.COOLDOWN = 1
    s.SCHEDULE.FACTOR = 0.5
    s.SCHEDULE.MAX_DECAY_STEP = 7

    s.PRE_VAL = False
    s.TO_VAL = True
    # every 10% training iterations completed, start an evaluation
    s.VAL_PERIOD = 3000
    s.CHECKPOINT_PERIOD = 5000

    s.USE_ATTN = False  # guided attention loss (TubeDETR comparison)
    s.SIGMA = 2.0  # std for quantized gaussian (KL-div loss)
    s.USE_AUX_LOSS = True  # auxiliary decoding losses (per layer)
    s.EOS_COEF = 0.1  # coeff for negative sample
    return s


def build_default_cfg() -> Cfg:
    """Construct default configuration tree.

    Returns an unfrozen CfgNode ready to be modified (merge_from_file/list).
    """
    root = Cfg()
    # top-level
    root.FROM_SCRATCH = True
    root.DATA_TRUNK = None
    root.OUTPUT_DIR = ''
    root.DATA_DIR = ''
    root.GLOVE_DIR = ''
    root.TENSORBOARD_DIR = ''

    # sections
    root.INPUT = _build_input_cfg()
    root.MODEL = _build_model_cfg()
    root.DATASET = _build_dataset_cfg()
    root.DATALOADER = _build_dataloader_cfg()
    root.SOLVER = _build_solver_cfg()
    return root


# Backward-compat alias (some code may import _C directly)
_C = build_default_cfg()