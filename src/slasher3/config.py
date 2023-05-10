from __future__ import annotations
from dataclasses import dataclass, field

import json
import os
import sys

from .log import logger

### config.py GLOBALS ###
ERROR_PARAMETERS_MSG = "Incorrect number of Parameters: python src/slasher3.py <slasher_name>"

class Config:
    def load_dict(self, configs : dict) -> Config:
        for config in configs:
            if hasattr(self, config):
                setattr(self, config, configs[config])
            else:
                raise(f"Unknown config {config}!")
        return self

class FolderConfig(Config):
    folder_list : list

    @property
    def check(self):
        for folder in self.folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

class ImageDirConfig(FolderConfig):
    xrays : str
    images : str
    masks : str
    to_mask : str
    input : str
    output : str
    def __init__(self, xrays : str, images : str, masks : str, to_mask : str, input : str, output : str):
        self.xrays = xrays
        self.images = os.path.join(self.xrays, images)
        self.masks = os.path.join(self.xrays, masks)
        self.to_mask = to_mask
        self.input = os.path.join(self.to_mask, input)
        self.output = os.path.join(self.to_mask, output)
        self.folder_list = [self.images, self.masks, self.input, self.output]

class AppDirConfig(FolderConfig):
    application : str
    source : str
    root : str
    configs : str
    models : str

    def __init__(self, application : str, configs : str, models : str):
        self.application = application
        self.source = os.path.dirname(self.application)
        self.root = os.path.dirname(self.source)
        self.configs = os.path.join(self.source, configs)
        self.models = os.path.join(self.source, models)
        self.folder_list = [self.application, self.source, self.root, self.configs, self.models]

@dataclass
class SlasherConfig(Config):
    # SOLVER META
    name : str = field(default=None) # SLASHER / TOTELES
    config_name : str = field(default=None) # xt64, xt128..

    # SOLVER PARAMETERS
    train : bool = field(default=None)
    local_or_cloud : str = field(default=None)
    override_device : bool = field(default=None)
    random_seed : int = field(default=None)

    # DATABASE PARAMETERS
    image_dir : ImageDirConfig = field(default=None)
    splitter : str = field(default=None)
    training_set : list[str] = field(default=None)
    use_dorothy : bool = field(default=None)

    # MODEL PARAMETERS
    max_images : int = field(default=None)
    output_width : int = field(default=None)
    batch_size : int = field(default=None)
    epochs : int = field(default=None)
    training_lr : int = field(default=None)
    patience : int = field(default=None)
    export_path : str = field(default=None)

### config.py INITIALIZATION ###

# INNER OBJECTS
# APP FOLDERS
app_folder = os.path.dirname(os.path.abspath(__file__)) # config.py is in slasher3 folder
app_dir = AppDirConfig(application=app_folder,
                       configs="configs",
                       models="slashers")
app_dir.check

# XRAY FOLDERS / API
xrays = os.path.join(app_dir.root, "xrays")
to_mask = os.path.join(app_dir.root, "to_mask")
image_dir = ImageDirConfig(xrays=xrays,
                           images="images",
                           masks="masks",
                           to_mask=to_mask,
                           input="input",
                           output="output")

image_dir.check

# READ COMMAND LINE PARAMETERS
if len(sys.argv) != 2:
    raise(ERROR_PARAMETERS_MSG)

# MODEL CONFIG FACTORY
config_name = sys.argv[1]
config_path = os.path.join(app_dir.configs, f"{config_name}.json")
slasher_config = None
if os.path.exists(config_path):
    logger.info(__file__, f"{config_name} config found!")
    with open(config_path, 'r') as config_json:
        configs = json.load(config_json)
    slasher_config = SlasherConfig().load_dict(configs)
else:
    raise(Exception(f"<slasher> model missing on config file {config_path}"))

slasher_config.image_dir = image_dir
slasher_config.config_name = config_name
slasher_config.export_path = os.path.join(app_dir.models, f"{config_name}")
slasher_config.net_save = None
if not os.path.exists(slasher_config.export_path):
    os.makedirs(slasher_config.export_path)
local_model_is_trained = False
if slasher_config.config_name:
    net_save = os.path.join(slasher_config.export_path, f"{slasher_config.config_name}.net")
    local_model_is_trained = os.path.exists(net_save)

if local_model_is_trained:
    logger.info(__file__, f"{config_name} model found!")
    slasher_config.net_save = net_save
else:
    logger.info(__file__, f"{config_name} model not found...")