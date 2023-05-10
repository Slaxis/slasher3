import os
from dataclasses import dataclass

import pandas
from pandas import DataFrame, read_csv
# # from matplotlib import pyplot
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from slasher3.log import logger
from slasher3.config import ImageDirConfig

from torchvision import transforms

@dataclass
class XRay:
    name : str
    dataset : str
    image_id : str
    image : Tensor
    mask : Tensor
    y : int

class XRays(dict[str:XRay]):
    training_set : list[str]
    width : int
    image_transforms : transforms.Compose
    mask_transforms : transforms.Compose
    def __init__(self, training_set, width, **kwargs):
        self.training_set = training_set
        self.width = width
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.width, self.width)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5,], std=[0.5,])
            ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize((self.width, self.width)),
            transforms.ToTensor(),
            ])
        
        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        pass

    def name(self, dataset : str, image_id : str) -> str:
        return dataset.lower() + "/" + image_id

    def check(self, dataset : str, image_id : str) -> bool:
        got_it = False
        xray = None
        try:
            xray = self.read(dataset, image_id)
            if xray:
                got_it = True
        except:
            pass

        return got_it

    def read(self, dataset : str, image_id : str) -> XRay:
        xray = None
        image_name = self.name(dataset, image_id)
        if image_name in self: # GOT ON CACHE
            return self[image_name]
        else: # NOT ON CACHE, READ FROM SOURCE
            xray = self._source(dataset, image_id)
            if xray:
                self[xray.name] = xray
                return self[xray.name]
            else:
                return None
        
    def _source(self, dataset : str, image_id : str) -> XRay:
        pass

class LocalXRays(XRays):
    local_folders : ImageDirConfig
    def initialize(self, **kwargs):
        self.local_folders = kwargs['local_folders']

    def _source(self, dataset : str, image_id : str) -> XRay:
        if dataset == 'input':
            image_path = os.path.join(self.local_folders.input, image_id + ".png")
            if os.path.exists(image_path):
                image_name = self.name(dataset, image_id)
                image = self.image_transforms(Image.open(image_path).convert('L'))
                y = -1
                return XRay(image_name, dataset, image_id, image, None, y)
            else:
                return None
        else:
            image_path = os.path.join(self.local_folders.images, dataset.lower(), image_id + ".png")
            mask_path = os.path.join(self.local_folders.masks, dataset.lower(), image_id + "_mask.png")
            if os.path.exists(image_path):
                image_name = self.name(dataset, image_id)
                image = self.image_transforms(Image.open(image_path).convert('L'))
                y = int(image_id[-1])
                if os.path.exists(mask_path): # CAN SOURCE IMAGE
                    mask = self.mask_transforms(Image.open(mask_path).convert('L'))
                    return XRay(image_name, dataset, image_id, image, mask, y)
                else:
                    # logger.info(LocalXRays.__init__.__qualname__, f"no mask for {image_name}!")
                    return None
            else:
                return None

class CloudXRays(XRays):
    pass

class XRayDataset(Dataset):
    xrays_reader : XRays
    xrays : DataFrame
    def __init__(self, xrays_reader : XRays, xrays : DataFrame):
        self.xrays_reader = xrays_reader
        self.xrays = xrays

    def __len__(self):
        return self.xrays.shape[0]

class XRayMask(XRayDataset):
    def __getitem__(self, index):
        xray_data = self.xrays.iloc[index]
        dataset = xray_data['run']
        image_id = xray_data['image_id']
        xray = self.xrays_reader.read(dataset, image_id)
        # x = xray.image*xray.mask
        # y = xray.y
        return xray.image, xray.mask

class XRayUnmasked(XRayDataset):
    def __getitem__(self, index):
        xray_data = self.xrays.iloc[index]
        dataset = xray_data['run']
        image_id = xray_data['image_id']
        xray = self.xrays_reader.read(dataset, image_id)
        # x = xray.image*xray.mask
        # y = xray.y
        return xray.image, xray.y #, xray.mask

class MaskedXRayClass(XRayDataset):
    def __getitem__(self, index):
        xray_data = self.xrays.iloc[index]
        dataset = xray_data['run']
        image_id = xray_data['image_id']
        xray = self.xrays_reader.read(dataset, image_id)
        x = xray.image*xray.mask
        y = xray.y
        return x, y

# Dataset Factory Through .sample()
class XRayDatasets:
    xray_reader : XRays
    dfolds : DataFrame # ['run', 'image_id', 'image_path', 'image_exists', 'mask_path', 'mask_exists', 'target']
    unique_images : DataFrame # ['run', 'image_id']
    def __init__(self, from_local_files : bool, local_folders : ImageDirConfig, splitter : str, training_set : list[str], width : int):
        self.from_local_files = from_local_files
        if self.from_local_files: # START LOCAL XRAY READER
            kwargs = {"local_folders" : local_folders}
            self.xray_reader = LocalXRays(training_set, width, **kwargs)
        else:
            self.xray_reader = CloudXRays() # NOT IMPLEMENTED

        splitter = splitter
        splitter_path = os.path.join(local_folders.xrays, splitter)
        logger.info(self.__init__.__qualname__, f"Reading local files from {splitter}")
        self.dfolds = read_csv(splitter_path, sep=',', decimal='.')
        self.dfolds.columns = ['xray_id', 'target', 'image_id', 'lps_path', 'image_hash', 'age', 'sex', 'comment', 'run', 'test', 'sort', 'dataset', 'type']

        # BUFFER ALL IMAGES
        logger.info(XRayDatasets.__init__.__qualname__, "Buffering all images on XRayReader...")
        self.unique_images = self.dfolds[['run', 'image_id']].value_counts().reset_index()[['run', 'image_id']]
        image_ok_list = []
        for run, image_id in [(x['run'], x['image_id']) for i, x in self.unique_images.iterrows()]:
            image_ok = self.xray_reader.check(run, image_id)
            image_ok_list.append({"run" : run, "image_id" : image_id, "ok" : image_ok})
        
        df_image_ok = pandas.DataFrame.from_dict(image_ok_list)
        self.unique_images = self.unique_images.set_index(['run', 'image_id']).join(df_image_ok.set_index(['run', 'image_id'])).reset_index()
        true_images = self.unique_images[self.unique_images.ok == True]
        self.dfolds = true_images.set_index(["run", "image_id"]).join(self.dfolds.set_index(["run", "image_id"])).reset_index()
        self.dfolds = self.dfolds.set_index(['sort', 'test', 'dataset', 'type', 'xray_id']).sort_index()
        self.dfolds = self.dfolds[(self.dfolds.run.str.lower().isin([x.lower() for x in training_set]))][['run', 'image_id', 'target']]

    @property
    def folds(self) -> int:
        return self.dfolds.reset_index()['sort'].unique()

    @property
    def test(self) -> int:
        return self.dfolds.reset_index()['test'].unique()

    def sample(self, kind : str, ix_sort:int = None, ix_test:int = None, sample_type: str = None, xray_kind: str = None, max_images : int = None, shuffle=True) -> XRayDataset:
        if sample_type in ["train", "val"]: # KILL MASKELESS ROWS
            xrays = self.dfolds.loc[(ix_sort, ix_test, sample_type, xray_kind)][['run', 'image_id', 'target']].copy()
        elif sample_type == "test":
            xrays_train = self.dfolds.loc[(ix_sort, ix_test, "train", xray_kind)][['run', 'image_id', 'target']].copy()
            xrays_test = self.dfolds.loc[(ix_sort, ix_test, "val", xray_kind)][['run', 'image_id', 'target']].copy()
            xrays = pandas.concat([xrays_train, xrays_test])
        
        elif sample_type == "input": # BUILD CUSTOM to_mask folder DataFrame
            xray_list = []

            if os.path.exists(self.xray_reader.local_folders.input):
                for xray_file in os.listdir(self.xray_reader.local_folders.input):
                    xray_list.append(("input", xray_file.removesuffix(".png"), -1))

                xrays = pandas.DataFrame.from_records(xray_list)
                xrays.columns = ['run', 'image_id', 'target']

        if max_images:
            xrays = xrays.sample(max_images)
        
        if shuffle:
            xrays = xrays.sample(frac=1)

        # if kind == "class":
        #     return MaskedXRayClass(self.xray_reader, xrays)
        if kind == "segment":
            return XRayMask(self.xray_reader, xrays)
        elif kind == "unmasked":
            return XRayUnmasked(self.xray_reader, xrays)

        return None