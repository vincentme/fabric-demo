# fabric-demo

在查找了大大小小众多pytorch训练框架后，发现并没有特别合适的。现在训练环境（cpu、gpu、ddp）、功能需求（混合精度、自动保存和恢复）越来越多，如果使用原始 pytorch 实现，那么大部分内容将是事务代码。而目前的框架如OpenMMLab和 lightning 等又追求大而全，做小修小改和对比试验容易，但是若大幅调整训练逻辑，就要深入各个 hook 里。现在另一个趋势是人们意识到小而精的辅助框架可能是研究者更需要的，如 huggingface 的 accelerate 和 lightning 的 fabric 正是其代表。

因此，我根据自己的习惯，主要基于lightning-fabric实现了一个简单灵活的pytorch训练框架/模板。共两百多行，不使用trainer，方便随意添加修改训练逻辑。

实现功能有：
*  训练设备切换和mixed precision混合精度：基于fabric，不修改运行代码而根据配置或自动选择cpu、gpu、tpu、mps等设备运行。支持DP、本机DDP和多节点DDP。支持mixed precision（32，16，16-mixed，bf16-mixed等）。
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

After searching for many pytorch training frameworks, large and small, I found that there was no particularly suitable one. There are now more and more training environments (cpu, gpu, ddp), functional requirements (mixed precision, automatic save and restore), and if you use the original pytorch implementation, most of it will be transactional code. Current frameworks such as OpenMMLab and Lightning are large and comprehensive. It is easy to make small modifications and comparative experiments. However, if the training logic is significantly adjusted, it is necessary to go deep into each hook. Another trend now is that people realize that small and sophisticated auxiliary frameworks may be what researchers need more, such as huggingface's accelerate and lightning's fabric are their representatives. 

Therefore, based on my own habits, I implemented a simple and flexible pytorch training framework/template mainly based on lightning-fabric. Only 200 lines in total, and no trainer is used, making it easy to add and modify training logic at will.  

The implemented functions are：
*  Training device switching and mixed precision: Based on fabric, it does not need to modify the running code, but selects cpu, gpu, tpu, mps and other devices to run according to the configuration file or command line options. Supports DP, single-node DDP and multi-node DDP. Support mixed precision(32, 16, 16-mixed, bf16-mixed, etc.). 
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
