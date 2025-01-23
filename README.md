# SKP: Rapid Prototyping and Iterative Experimentation

This code repository is what I use to train deep learning computer vision models. It uses [PyTorch Lightning](https://lightning.ai/pytorch-lightning)
for training and validation and [Neptune](https://neptune.ai) for logging. The goal of this codebase is to enable rapid prototyping and experimentation,
and the design choices I made were largely based on my experiences competing in Kaggle competitions.

## Installation
`git clone https://github.com/i-pan/SKP ; cd SKP ; pip install -e .[full]`

## Configs

Configuration files live as Python files in `skp/configs`. I use subdirectories to separate config files related to different projects. 
I have found Python files to be the most flexible as compared to other options such as YAML. In particular, I like being able to define the
augmentations directly in the configuration file. 

The main entry point is `skp/train.py`. To run a specific training configuration, use the command line:

`python skp/train.py boneage.cfg_baseline`

Here, `boneage` is the subdirectory within `skp/configs` where config files for this project live. `cfg_baseline` is the name of the Python
configuration file within this directory. Note that the `.py` extension is omitted from the command line argument.

Attributes in the configuration files can be overwritten in the command line. For example:

`python skp/train.py boneage.cfg_baseline --batch_size 16`

For `bool` values, you must specify `True` or `False` as the value (e.g., `--load_pretrained False`). For dictionaries, you can also overwrite specific keys using `--{dict}.{key} {value}`, for example:

`python skp/train.py boneage.cfg_baseline --optimizer_params.lr 0.001`

Nested dictionaries are not supported.

Certain attributes should not be overwritten, namely any attribute that is later referenced in the file. For example, `cfg.image_height` and
`cfg.image_width` are almost always referenced within the resize transforms in `cfg.train_transforms` and `cfg.val_transforms`. Overwriting
these attributes in the command line will only change those specific attributes. However, any later references to these attributes will not
be changed.

Also, only attributes which are already defined in the configuration file can be overwritten. This encourages you to be explicit in defining
the all the relevant attributes for your training run. There are some exceptions, for example if you wish to enable exponential moving
averaging (EMA), which are set in the base `Config` class (found in `skp.configs.__init__.py`):

`python skp/train.py boneage.cfg_baseline --ema.on True --ema.update_after_step 100 --ema.decay 0.999 --ema.switch_ema True`

## Models

Model achitectures lives as Python files in `skp/models`. Thera are subdirectories defining the type of model architecture (e.g., classification,
segmentation, etc.). Each model architecture is defined in a separate Python file. The main model class should be named `Net` and should only
accept the config as an argument. Also, the `forward` method takes as primary input a dictionary and produces a dictionary as output. 
This is to provide maximum flexibility. The model is simply a PyTorch `nn.Module`. In the config file, the model is set as `cfg.model`. 
For example, if you want to use `skp/models/classification/net2d.py`, you would set `cfg.model = classification.net2d`. 

## Datasets

Datasets follow a similar structure to the above and live in `skp/datasets`. Each dataset is defined in a separate Python file, and the main dataset class is named `Dataset`. It takes only the config file and mode (i.e., train, val, inference) as arguments. Some datasets live directly under `skp/datasets`, while others
are project-specific and live within subdirectories. In the config file, the dataset is set as `cfg.dataset`. Datasets will return an input dictionary.

## Losses

Losses live in `skp/losses`, and individual loss functions are defined as separate classes with several Python files, depending on the task,
for example, `classification.py` and `segmentation.py`. The `forward` method takes as primary input the prediction dictionary (typically output from model)
and ground truth dictionary (dataset output). 

## Metrics

Metrics follow a similar structure to losses and are built on `torchmetrics.Metric`, which is a flexible class that also handles distributed validation.

## Tasks

Tasks are `LightningModule` classes where you can define various training and validation hooks. Defining a new task is helpful if your model's training
or validation steps deviate from the typical setup (for example, if you are using multiple optimizers). Custom samplers are also defined here in
`skp/tasks/samplers.py`.

## Optimization

Custom optimizers and learning rate schedulers can be added to `skp/optim`. See `skp/optim/adamp.py` and `skp/optim/linear_warmup_cosine_annealing.py`
for examples. 

## Toolbox

Miscellaneous assortment of oft-used functions and classes are available in `skp/toolbox`. 
