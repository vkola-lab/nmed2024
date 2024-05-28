__version__ = '0.0.1'

from . import nn
from . import model

# # load pretrained transformer
# pretrained_transformer = model.Transformer.from_ckpt('{}/ckpt/ckpt.pt'.format(__path__[0]))
# from . import shap_adrd
# from .model import DynamicCalibratedClassifier
# from .model import StaticCalibratedClassifier

# load fitted transformer and calibrated wrapper
# try:
#     fitted_resnet3d = model.CNNResNet3DWithLinearClassifier.from_ckpt('{}/ckpt/ckpt_img_072523.pt'.format(__path__[0]))
#     fitted_calibrated_classifier_nonimg = StaticCalibratedClassifier.from_ckpt(
#         filepath_state_dict = '{}/ckpt/static_calibrated_classifier_073023.pkl'.format(__path__[0]),
#         filepath_wrapped_model = '{}/ckpt/ckpt_080823.pt'.format(__path__[0]),
#     )
#     fitted_transformer_nonimg = fitted_calibrated_classifier_nonimg.model
#     shap_explainer = shap_adrd.SamplingExplainer(fitted_transformer_nonimg)
# except:
#     print('Fail to load checkpoints.')
