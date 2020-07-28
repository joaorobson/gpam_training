"""
Module to facilitate the integration of a sklearn training pipeline into a deploy and retraining system
"""

from .multilabel_training import MultilabelTraining
from .binary_training import BinaryTraining
from .metrics import *

__version__ = "0.0.15"
