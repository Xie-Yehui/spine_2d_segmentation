import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.dataset import ISBI_Loader
import network.archs
import utils.losses
# from dataset import Dataset
from utils.metrics import iou_score
from utils.utils import AverageMeter, str2bool