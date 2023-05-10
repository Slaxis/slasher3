import torch
from slasher3.log import logger

def get_device(override_device):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if override_device:
        device_name = "cpu"
    device = torch.device(device_name)
    logger.info(get_device.__qualname__,"DEVICE: {}".format(device))
    return device