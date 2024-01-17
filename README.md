# fabric-demo
A simple and flexible pytorch training framework/template based on lightning-fabric

After searching for many pytorch training frameworks, large and small, based on my own habits, I implemented a simple and flexible pytorch training framework/template mainly based on lightning-fabric. Only 200 lines in total, and no trainer is used, making it easy to add and modify training logic at will. The implemented functions are：

*   Training device switching: Based on fabric, it does not need to modify the running code, but selects cpu, gpu, tpu, mps and other devices to run according to the configuration file or command line options. Supports DP, single-node DDP and multi-node DDP. 
*   Automatic batch size and learning rate adjustment: The actual batch size is automatically calculated based on the set effective batch size and world size (total number of devices) to ensure consistent results under different numbers of devices. There are two learning rate adjustment methods: linear and square root. 
*   Experiment management
    *   config: Default configuration file, supports command line option override. Automatically back up final configuration. 
    *   logger: Supports three loggers currently supported by fabric: csv, tensorboard and wandb. You can choose to use a single or multiple loggers. 
    *   out\_dir experiment folder: Add the date and time before out\_dir to generate the final experiment folder name. Therefore, a different experiment folder will be created for each run. If root\_dir is set, the final experimental folder out\_dir=root\_dir/out\_dir. 
    *   Checkpoint: According to checkpoint\_frequency, state (including model, optimizer, config, etc.) is automatically saved every few epochs. You can set num\_checkpoint\_keep to automatically retain the most recent checkpoints. 
    *   resume: Automatically find the latest checkpoint in the out\_dir/checkpoints folder and resume training.

Dependency
----------

Install lightning, torchmetrics, etc.

```text-x-sh
pip install -U lightning torchmetrics simple_parsing tensorboard wandb
```

Run
---

Training by run train.py. You can override the option and add comments. 

```text-x-python
python train.py --log_per_epoch 4 --logger_name  tensorboard csv  --checkpoint_frequency 5  --comment "concise comment for current run"
```

Testing by run test.py. out\_dir should be specified(data and time included).  

```text-x-sh
python test.py --out_dir 20240117-133637-train_out
```
