"""
Module to facilitate the integration of a sklearn training pipeline into a deploy and retraining system
"""

from .multilabel_training import MultilabelTraining
from .metrics import *
from .cnn_pecas_model import PecasModel

__version__ = "0.0.5"
