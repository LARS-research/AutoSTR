from __future__ import absolute_import

from .resnet_aster import *
# from .OneShotSinglePath_MobileOps import SuperNet_MBConvs, SuperNet_MBConvs_Compact
from .proxyless import ProxylessBackbone, CompactRecBackbone
from .darts import DartsBackbone
from .autodeeplab import AutoDeepLabBackbone

__factory = {
  'ResNet_ASTER': ResNet_ASTER,
  'Tiny_ResNet_ASTER': Tiny_ResNet_ASTER,
  'ProxylessBackbone': ProxylessBackbone,
  'CompactRecBackbone': CompactRecBackbone,
  'DartsBackbone': DartsBackbone,
  "AutoDeepLabBackbone": AutoDeepLabBackbone
  # 'SuperNet_MBConvs': SuperNet_MBConvs,
  # 'SuperNet_MBConvs_Compact': SuperNet_MBConvs_Compact
}

def names():
  return sorted(__factory.keys())


def create(name, *args, **kwargs):
  """Create a model instance.
  
  Parameters
  ----------
  name: str
    Model name. One of __factory
  pretrained: bool, optional
    If True, will use ImageNet pretrained model. Default: True
  num_classes: int, optional
    If positive, will change the original classifier the fit the new classifier with num_classes. Default: True
  with_words: bool, optional
    If True, the input of this model is the combination of image and word. Default: False
  """
  if name not in __factory:
    raise KeyError('Unknown model:', name)
  return __factory[name](*args, **kwargs)
