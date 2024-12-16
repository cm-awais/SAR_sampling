from m_models import *
from m_utils import *
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import warnings
import random
import torch.backends.cudnn as cudnn

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # for multi-GPU setups

# Disable CuDNN heuristics
cudnn.benchmark = False
cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes = 9
lf = nn.CrossEntropyLoss()
epochs = 30
l_rate = 0.001
batch_size = 32

models = {"NVIT": ViTModel(classes, frozen=False),
    "VIT": ViTModel(classes, frozen=True),
    "VGG": VGGModel(classes),
    "Fine_VGG": FineTunedVGG(classes),
    "ResNet": ResNetModel(classes),
    "Fine_Resnet": FineTunedResNet(classes)
    }

c_datasets_b = {"Fusar": get_loaders_b(1, batch_size=batch_size)}


train_save(c_datasets_b=c_datasets_b, models=models, l_rate=l_rate, device=device, epochs=epochs, classes=classes, lf=lf)