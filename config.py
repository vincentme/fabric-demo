from dataclasses import dataclass
from typing import List

from simple_parsing.helpers import Serializable
from simple_parsing import choice, field


@dataclass
class TrainConfig(Serializable):
    """Help string for training config"""
    accelerator: str = "auto"
    devices: str = "auto"
    project: str = 'MNIST-cls'
    comment: str = None
    root_dir: str = None # where to store out_dir
    out_dir: str = 'train_out' # output directory name
    logger_name: List[str] = field(default_factory=list)
    wandb_name: str = None
    resume: str = None
    seed: int = 42 # random seed
    fast_run: bool = False
    precision: str = 'auto' # "32-true", "16-mixed", "bf16-mixed", etc. 
    
    ## data
    dataset_path: str = 'dataset'
    eff_batch_size: int = 1024 # effective batch_size = num_node*num_process_per_node*batch_size_per_process = world_size*batch_size_per_process
    batch_size: int = None
    
    ## train
    epochs: int = 200 # number of epochs to train
    warmup: int = 10 # warmup beta and lr
    base_lr: float = 1e-3 # base learning rate. Using linear law, lr = eff_batch_size/base_bs*base_lr
    base_batch_size = 64 # base batch_size
    lr: float = None
    lr_scaling_rule: str = choice("sqrt", "linear", default="linear") # learning rate should scale with batch size, choose from linear or sqrt(square root)
    gamma: float = 0.95 # Learning rate step gamma

    ## logging and checking
    log_interval: int = 100 # how many batches to wait before logging training status
    log_per_epoch: int = None # number of log for each epoch. If set will override log_interval
    checkpoint_frequency: int = None # number of epoch interval to save the checkpoint
    num_checkpoint_keep: int = 3 # set to None to disable this behavior


@dataclass
class TestConfig(Serializable):
    """Help string for testing config"""
    accelerator: str = "auto"
    devices: str = "auto"
    comment: str = None
    root_dir: str = None
    out_dir: str = 'test_out'
    checkpoint_file: str = 'final_checkpoint.ckpt'
    seed: int = 42 # random seed
    
    ## data
    dataset_path: str = 'dataset'
    eff_batch_size: int = 1024 # effective batch_size = num_node*num_process_per_node*batch_size_per_process = world_size*batch_size_per_process
    batch_size: int = None