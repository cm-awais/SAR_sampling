from m_models import *
from dis_utils import *
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import warnings
import random
import torch.backends.cudnn as cudnn
import time

# t0 = time.time()

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

# 224 for fusar_ship and ViT
# 64 for opensar ship

image_size = 224

transform_train_opensar = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

image_size = 224
transform_train_fusar = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# classes = 9
lf = nn.CrossEntropyLoss()
epochs = 30
l_rate = 0.001
batch_size = 32

fusar_2_classes = ["Fishing", "Cargo"]  # <800
fusar_4_classes = ["Fishing", "Cargo", "Bulk", "Tanker"]  # <500
fusar_9_classes = ["Fishing", "Cargo", "Bulk", "Tanker", "Tug", "Container", "Passenger", "GeneralCargo", "Dredging"]  # 500

opensar_2_classes = ["Tanker", "Cargo"]  # <2000
opensar_4_classes = ["Fishing", "Cargo", "Dredging", "Tanker"] # <200
opensar_6_classes = ["Fishing", "Cargo", "Dredging", "Tanker", "Tug", "Passenger"]  # <200

opensar_path = "/home/si-lab/Desktop/Projects/1-thesis/Ship-Classification-SAR-DL/experiments/curriculum/c_opensar_splited"
fusar_path = "/home/si-lab/Desktop/Projects/1-thesis/Ship-Classification-SAR-DL/experiments/curriculum/c_fusar_splited"

# c_datasets_b = {
#     "Fusar_2": get_loaders_b(fusar_path, data=1, classes_l=fusar_2_classes, batch_size=batch_size),
#     "Fusar_4": get_loaders_b(fusar_path, data=1, classes_l=fusar_4_classes, batch_size=batch_size),
#     "Fusar_9": get_loaders_b(fusar_path, data=1, classes_l=fusar_9_classes, batch_size=batch_size),
#     "Opensar_2": get_loaders_b(opensar_path, data=0, classes_l=opensar_2_classes, batch_size=batch_size),
#     "Opensar_4": get_loaders_b(opensar_path, data=0, classes_l=opensar_4_classes, batch_size=batch_size),
#     "Opensar_6": get_loaders_b(opensar_path, data=0, classes_l=opensar_6_classes, batch_size=batch_size)
#     }
t0 = time.time()
data_dict = {
    "Fusar_2": [get_loaders_b("c_fusar_2", data=1, classes_l=fusar_2_classes, transform_train=transform_train_fusar, batch_size=batch_size, m2m=True), 2, 800],
    "Fusar_4": [get_loaders_b("c_fusar_4", data=1, classes_l=fusar_4_classes, transform_train=transform_train_fusar, batch_size=batch_size, m2m=True), 4, 500],
    "Fusar_9": [get_loaders_b("c_fusar_9", data=1, classes_l=fusar_9_classes, transform_train=transform_train_fusar, batch_size=batch_size, m2m=True), 9, 500],
    "Opensar_2": [get_loaders_b("c_opensar_2", data=0, classes_l=opensar_2_classes, batch_size=batch_size, transform_train=transform_train_opensar, m2m=True), 2, 3000],
    "Opensar_4": [get_loaders_b("c_opensar_4", data=0, classes_l=opensar_4_classes, batch_size=batch_size, transform_train=transform_train_opensar, m2m=True), 4, 250],
    "Opensar_6": [get_loaders_b("c_opensar_6", data=0, classes_l=opensar_6_classes, batch_size=batch_size, transform_train=transform_train_opensar, m2m=True), 6, 250]
}
t1 = time.time()
total_loaders = t1-t0

train_save_b(data_dict=data_dict, l_rate=l_rate, device=device, epochs=epochs, lf=lf, m2m=True)
print("Time for DataLoaders for M2m", total_loaders)
t2 = time.time()
total_train = t2-t1
print("Time for training for M2m", total_train)
total = t2-t0
print("Time for M2m", total)
