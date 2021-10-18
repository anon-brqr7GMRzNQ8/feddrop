from .base import LayerDropout
from .cnns.lenet import LeNet, SVHNNet
from .cnns.fashion import FashionNet
from .cnns.mobilenet_v2 import MobileNetV2
from .rnns.transformer import TransformerModel
from .cnns.cifarnet import MCifarnet, tfCifarnet
from .synthetic_nn import SyntheticNN
from .cnns.vgg import vgg9, vgg9_bn

factory = {
    'synthetic_nn': SyntheticNN,
    'lenet': LeNet,
    'svhnnet': SVHNNet,
    'fashionnet': FashionNet,
    'mobilenet_v2': MobileNetV2,
    'transformer': TransformerModel,
    'mcifarnet': MCifarnet,
    'vgg9': vgg9,
    'vgg9_bn': vgg9_bn,
    'tfcifarnet': tfCifarnet,
}
