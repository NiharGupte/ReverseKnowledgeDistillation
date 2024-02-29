import torch
import os
import wandb
from torch.utils.data import DataLoader

from common.train_util_original import train_model
from dataset.retina_dataset import RetinaDataset
from model.LKUNET import SuperRetina

import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
import warnings
import dill
warnings.filterwarnings('ignore')

config_path = './config/train.yaml'

if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config File doesn't Exist")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
    
assert 'MODEL' in config
assert 'PKE' in config
assert 'DATASET' in config
assert 'VALUE_MAP' in config
train_config = {**config['MODEL'], **config['PKE'], **config['DATASET'], **config['VALUE_MAP']}

batch_size = train_config['batch_size']
num_epoch = train_config['num_epoch']
device = train_config['device']
device = torch.device(device if torch.cuda.is_available() else "cpu")

dataset_path = train_config['dataset_path']
data_shape = (train_config['model_image_height'], train_config['model_image_width'])

train_split_file = train_config['train_split_file']
val_split_file = train_config['val_split_file']
auxiliary = train_config['auxiliary']
train_set = RetinaDataset(dataset_path, split_file=train_split_file,
                          is_train=True, data_shape=data_shape, auxiliary=auxiliary)
val_set = RetinaDataset(dataset_path, split_file=val_split_file, is_train=False, data_shape=data_shape)

load_pre_trained_model = train_config['load_pre_trained_model']
pretrained_path = train_config['pretrained_path']

model = SuperRetina(train_config, device=device)

# pt_weights = torch.load("Final_tests/model_bestValRMSE.pth.tar",  pickle_module = dill)

# model.load_from(pt_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Trainable Total Parameters : ", count_parameters(model))

# exit()

if load_pre_trained_model:
    if not os.path.exists(pretrained_path):
        raise Exception('Pretrained model doesn\'t exist')
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['net'])

optimizer = optim.Adam(model.parameters(), lr=1e-4)

dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)
    }

with torch.autograd.set_detect_anomaly(True):
    model = train_model(model, optimizer, dataloaders, device, num_epochs=num_epoch, train_config=train_config)

wandb.finish()


