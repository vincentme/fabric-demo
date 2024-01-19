import os, time, datetime
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from lightning.fabric import Fabric, seed_everything
import lightning as L

from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import RunningMean, MeanMetric

from simple_parsing import ArgumentParser
from config import TrainConfig

from dataset import get_dataloaders
from utils import num_parameters, choose_logger, get_checkpoint_files

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


def lr_lambda(epoch, warmup, gamma):
    if epoch <= warmup:
        return epoch/warmup+1e-3
    else:
        return gamma**(epoch-warmup)

def main(fabric, config):
    seed_everything(config.seed, workers = True)
    
    train_dataloader, val_dataloader = get_dataloaders(fabric, config)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    num_iter_per_epoch = len(train_dataloader)
    
    # model will be created directly on device
    with fabric.init_module():
        model = Net()
    # model = torch.compile(model)
    
    if config.lr_scaling_rule == 'linear':
        config.lr = config.eff_batch_size/config.base_batch_size*config.base_lr
    elif config.lr_scaling_rule == 'sqrt':
        config.lr = (config.eff_batch_size/config.base_batch_size)**0.5*config.base_lr

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum = 0.9)
    model, optimizer = fabric.setup(model, optimizer)
    
    scheduler = LambdaLR(optimizer, partial(lr_lambda, warmup=config.warmup, gamma = config.gamma))

    state = {"model": model, "optimizer": optimizer, "config": config, "iter_num": 0, 'epoch_num':0}
    
    checkpoint_files = get_checkpoint_files(config.checkpoint_dir)
    if config.resume and len(checkpoint_files) > 0:
        resume_file = max(checkpoint_files, key=(lambda p: int(p.split('-')[-1].split('.')[0])))
        fabric.load(resume_file, state)
        scheduler = LambdaLR(optimizer, partial(lr_lambda, warmup=config.warmup, gamma = config.gamma), last_epoch = state['epoch_num'])
        fabric.print(f"Resuming training from {resume_file}")
        state['epoch_num'] += 1
    else:
        # if do not resume, remove all existed checkpoint
        if fabric.global_rank == 0:
            for checkpoint_file in checkpoint_files:
                os.remove(checkpoint_file)
        fabric.barrier()

    # if set log_per_epoch, then change log_interval according to it
    if config.log_per_epoch:
        config.log_interval = max(round(num_iter_per_epoch/config.log_per_epoch), 1)
        fabric.print(f'set log_interval to {config.log_interval}')
        
    # save wandb auto-generated run name in config
    for logger in fabric.loggers:
        if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
            config.wandb_name = logger.experiment.name
        
    # print and log config
    config.save(os.path.join(config.out_dir, "config.yaml"))
    fabric.print('-----Final Config-----')
    fabric.print(config.dumps_yaml(), end='')
    fabric.print('----------------------')
    fabric.print(f"total parameters: {num_parameters(model):,}")
    fabric.print(f"accelerator: {fabric.accelerator}")
    fabric.print(f"strategy: {fabric.strategy}")
    fabric.print('----------------------')
    for logger in fabric.loggers:
        if isinstance(logger, (L.fabric.loggers.TensorBoardLogger, L.pytorch.loggers.wandb.WandbLogger)):
            logger.log_hyperparams(config.to_dict())
    
    # use torchmetrics to track loss and compute accuracy
    with fabric.init_module():
        train_mean_loss = MeanMetric()
        train_running_mean_loss = RunningMean()
        val_mean_loss = MeanMetric()
        val_acc = Accuracy(task="multiclass", num_classes=10)
    
    fabric.barrier()
    train_t0 = time.perf_counter()

    for epoch in range(state['epoch_num'], config.epochs):
        state['epoch_num'] = epoch
        
        # training
        model.train()
        for batch_idx, batch_data in enumerate(train_dataloader):
            iter_t0 = time.perf_counter()
            
            data, target = batch_data
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            fabric.backward(loss)
            optimizer.step()
            
            train_mean_loss(loss)
            train_running_mean_loss(loss)
            
            # printing during training
            if state['iter_num'] % config.log_interval == 0:
                metrics = {
                    "loss": train_running_mean_loss.compute().item(),
                    "iter_num": state["iter_num"],
                    "epoch_num": epoch,
                    "iter_time": time.perf_counter() - iter_t0,
                }
                fabric.log_dict(metrics, step=state["iter_num"])
                iter_per = round(batch_idx/num_iter_per_epoch*100) # percentage of iteration progress
                fabric.print(f"epoch {epoch} iter {state['iter_num']} {iter_per:d}%: running_mean_loss={metrics['loss']:.2e} iter_time= {metrics['iter_time'] * 1000:.1e}ms"  )

            state["iter_num"] += 1

        cur_train_mean_loss = train_mean_loss.compute().item()
        train_mean_loss.reset()

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
        val_mean_loss.reset()
        val_acc.reset()
        
        # printing for each epoch
        fabric.print(f"epoch {epoch}: train(loss={cur_train_mean_loss:.2e}")
        fabric.print(f"validation(loss={cur_val_mean_loss:.2e} accuracy={cur_val_acc*100:.1f}% time= {(time.perf_counter() - val_t0):.2f}s)")
        epoch_metrics = {"train_loss":cur_train_mean_loss, "val_loss": cur_val_mean_loss, 'val_acc':cur_val_acc, 'lr':optimizer.param_groups[0]['lr']}
        fabric.log_dict(epoch_metrics, step=state["iter_num"])

        # checkpointing
        if config.checkpoint_frequency is not None and epoch % config.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"epoch-{epoch:05d}.ckpt")
            fabric.print(f"Saving checkpoint to {checkpoint_path}")
            fabric.save(checkpoint_path, state)
            fabric.barrier()
            
            # retain most recent num_checkpoint_keep checkpoints
            if fabric.global_rank == 0:
                if config.num_checkpoint_keep is not None:
                    candidate_checkpoint_files = get_checkpoint_files(config.checkpoint_dir)
                    num_checkpoint_files = len(candidate_checkpoint_files)
                    
                    if num_checkpoint_files > config.num_checkpoint_keep:
                        for i in range(num_checkpoint_files-config.num_checkpoint_keep):
                            os.remove(candidate_checkpoint_files[i])
            fabric.barrier()

        fabric.print('----------------------')
        scheduler.step()

    fabric.print(f"Training time: {(time.perf_counter()-train_t0):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    fabric.save(os.path.join(config.out_dir,"final_model.pt"), model.state_dict())
    fabric.save(os.path.join(config.out_dir,"final_checkpoint.ckpt"), state)

    for logger in fabric.loggers:
        if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
            logger.experiment.finish(exit_code = 0)
        logger.finalize('success')


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train")
    config = parser.parse_args().train
    
    if config.resume:
        config.out_dir = config.resume
    else:
        # prefix date and time in out_dir
        config.out_dir = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{config.out_dir}"
        
    # add root_dir to out_dir if set
    if config.root_dir:
        config.out_dir = os.path.join(config.root_dir, config.out_dir)
    
    print('------Load Config-----')
    print(config.dumps_yaml(), end='')
    print('----------------------')

    log_dir = os.path.join(config.out_dir, 'logs')
    loggers = [choose_logger(logger_name, log_dir = log_dir, project = config.project, comment = config.comment) for logger_name in config.logger_name]

    # create output folders
    config.checkpoint_dir = os.path.join(config.out_dir, 'checkpoints')
    os.makedirs(config.out_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    for logger in loggers:
        if isinstance(logger, L.fabric.loggers.CSVLogger):
            os.makedirs(os.path.join(log_dir, 'csv'), exist_ok=True)
        elif isinstance(logger, L.fabric.loggers.TensorBoardLogger):
            os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
        elif isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
            os.makedirs(os.path.join(log_dir, 'wandb'), exist_ok=True)
            
    fabric = Fabric(accelerator=config.accelerator, devices=config.devices, loggers = loggers, precision = config.precision)
    # fabric = Fabric(accelerator='cpu', devices=12, loggers = loggers)
    # fabric = Fabric(accelerator='gpu', devices=2, loggers = loggers)
    
    if isinstance(fabric.accelerator, L.fabric.accelerators.CUDAAccelerator):
        fabric.print('set float32 matmul precision to high')
        torch.set_float32_matmul_precision('high')

    try:
        fabric.launch(main, config) # for launching fabric in notebook or ipython consol, see https://lightning.ai/docs/fabric/stable/fundamentals/notebooks.html
    except KeyboardInterrupt:
        for logger in fabric.loggers:
            if isinstance(logger, L.pytorch.loggers.wandb.WandbLogger):
                logger.experiment.finish(exit_code = 1)
            logger.finalize('aborted')
