import os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.fabric import Fabric

from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import MeanMetric

from simple_parsing import ArgumentParser
from config import TestConfig

from dataset import get_dataloaders
from utils import num_parameters

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv1 = nn.Conv2d(1, 256, 3, 1)
        self.conv2 = nn.Conv2d(256, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main(fabric, config):
    fabric.seed_everything(config.seed, workers = True)
    
    train_dataloader, val_dataloader = get_dataloaders(fabric, config)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    
    with fabric.init_module():
        # models will be created directly on device
        model = Net()
    fabric.print(f"Total parameters: {num_parameters(model):,}")
    # model = torch.compile(model)
    
    model = fabric.setup(model)
    
    state = {"model": model, "config": config, "iter_num": 0, 'epoch_num':0}
    
    checkpoint_file = os.path.join(config.out_dir, config.checkpoint_file)
    fabric.print(f"Resuming training from {checkpoint_file}")
    if checkpoint_file.endswith('.ckpt'):
        fabric.load(checkpoint_file, state)
    elif checkpoint_file.endswith('.pt'):
        fabric.load(checkpoint_file, model)
        
    # print and log config
    config.save(os.path.join(config.test_out_dir, "config.yaml"))
    fabric.print('-----Final Config-----')
    fabric.print(config.dumps_yaml(), end='')
    fabric.print('----------------------')
    fabric.print(f"total parameters: {num_parameters(model):,}")
    fabric.print(f"accelerator: {fabric.accelerator}")
    fabric.print(f"strategy: {fabric.strategy}")
    fabric.print('----------------------')
    
    # use torchmetrics instead of manually compute the accuracy
    val_mean_loss = MeanMetric().to(fabric.device)
    val_acc = Accuracy(task="multiclass", num_classes=10).to(fabric.device)
    
    # validation
    model.eval()
    with torch.inference_mode():
        val_t0 = time.perf_counter()
        for batch_data in val_dataloader:
            data, target = batch_data
            
            output = model(data)
            loss = F.nll_loss(output, target)
            
            val_mean_loss(loss)
            val_acc(output, target)

    cur_val_acc = val_acc.compute().item()
    cur_val_mean_loss = val_mean_loss.compute().item()
    
    fabric.print(f"validation(loss={cur_val_mean_loss:.2e} accuracy={cur_val_acc*100:.1f}% time= {(time.perf_counter() - val_t0):.2f}s)")

    fabric.print('----------------------')
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_arguments(TestConfig, dest="test")
    config = parser.parse_args().test
    
    # add root_dir to out_dir if set
    if config.root_dir:
        config.out_dir = os.path.join(config.root_dir, config.out_dir)
    
    config.test_out_dir = os.path.join(config.out_dir, 'test_out')
    os.makedirs(config.test_out_dir, exist_ok=True)
    
    print('------Load Config-----')
    print(config.dumps_yaml(), end='')
    print('----------------------')
            
    fabric = Fabric(accelerator=config.accelerator, devices=config.devices)
    # fabric = Fabric(accelerator='gpu', devices = 2)
    
    fabric.launch(main, config)
