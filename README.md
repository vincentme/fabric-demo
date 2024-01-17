# fabric-demo

在查找了大大小小众多pytorch训练框架后，根据自己的习惯，主要基于lightning-fabric实现了一个简单灵活的pytorch训练框架/模板。共两百多行，不使用trainer，方便随意添加修改训练逻辑。实现功能有：

*   训练设备切换：基于fabric，不修改运行代码而根据配置或自动选择cpu、gpu、tpu、mps等设备运行。支持DP、本机DDP和多节点DDP。
*   自动batch size批大小和learning rate学习率调整：实际batch size根据设定的effective batch size和world size（总设备数）自动计算，保证不同设备数量下得到一致的结果。learning rate调整方案有linear线性和square root平方根两种。
*   实验管理
    *   config配置：默认配置文件，支持命令行选项覆盖。自动备份最终配置。
    *   logger记录器：支持fabric目前支持的三个logger：csv、tensorboard和wandb。可以选择使用单个或多个logger。
    *   out\_dir实验文件夹：在out\_dir前添加日期时间生成最终实验文件夹名。因此每次运行都会创建不同的实验文件夹。如设置了root\_dir，则最终实验文件夹out\_dir=root\_dir/out\_dir。
    *   checkpoint检查点：根据checkpoint\_frequency，每隔数个epoch自动保存state（包含model、optimizer和config等）。可以设置num\_checkpoint\_keep，自动保留最近数个checkpoints。
    *   resume恢复：在out\_dir/checkpoints文件夹中自动找到最新的checkpoint并恢复训练。

依赖
--

安装lightning、torchmetrics等

```text-x-sh
pip install -U lightning torchmetrics simple_parsing tensorboard wandb
```

运行
--

训练运行train.py，可在命令行覆盖参数和设置comment等

```text-x-python
python train.py --log_per_epoch 4 --logger_name  tensorboard csv  --checkpoint_frequency 5  --comment "concise comment for current run"
```

测试运行test.py，须传入先前的out\_dir（包含日期时间）

```text-x-sh
python test.py --out_dir train_out
```

* * *

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
