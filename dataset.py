import torch.utils.data
from torchvision import datasets, transforms
import multiprocessing

num_core = multiprocessing.cpu_count()

def get_dataloaders(fabric, config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    with fabric.rank_zero_first():  # set `local=True` if your filesystem is not shared between machines
        train_dataset = datasets.MNIST(config.dataset_path, download=fabric.is_global_zero, train=True, transform=transform)
        val_dataset = datasets.MNIST(config.dataset_path, download=fabric.is_global_zero, train=False, transform=transform)
        
    if hasattr(config, 'fast_run') and config.fast_run:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(config.eff_batch_size*4)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(config.eff_batch_size*2)))
    
    config.batch_size = round(config.eff_batch_size/fabric.world_size) # using the effective batch_size to calculate the batch_size per gpu
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,  batch_size=config.batch_size, shuffle=True, drop_last = True, num_workers = min(fabric.world_size*2, num_core), pin_memory = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, drop_last = True, num_workers = min(fabric.world_size*2, num_core), pin_memory = True)


    return train_dataloader, val_dataloader