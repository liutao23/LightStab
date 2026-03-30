from easydict import EasyDict as edict
__C = edict()
cfg = __C

__C.MODEL = edict()

__C.MODEL.NUM = 100
__C.MODEL.padding = 1
__C.MODEL.dilation = 1
__C.MODEL.T = 30
__C.MODEL.HEIGHT = 480
__C.MODEL.WIDTH = 640
__C.MODEL.PIXELS = 32
__C.MODEL.TOPK = 512
__C.MODEL.THRESHOLDPOINT = 102
__C.MODEL.RADIUS = 200
__C.MODEL.FLOWC = 20
__C.Threshold = edict()
__C.Threshold.MANG = 2
__C.Threshold.ROT = 5