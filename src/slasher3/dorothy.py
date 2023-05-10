import json
import os

import requests
from pandas import DataFrame

from .log import logger

LPS_PATH = "https://dorothy-image.lps.ufrj.br/images/?search="

class Dorothy:
    token : str

    def __init__(self, token : str = 'b16fe0fc92088c4840a98160f3848839e68b1148'):
        self.token = token

    def access(self, dataset : str) -> tuple[dict, dict]: # data / header
        header = { "Authorization": 'Token '+ str(self.token)}
        response = requests.get(f"{LPS_PATH}{dataset}", headers=header)
        data = json.loads(response.content)
        return data, header
    
    def download_to(self, download_dir, dataset, data, metadata, header) -> bool: # DONWLOAD OK
        try:
            if os.path.exists(download_dir):
                for img in data:
                    file = open(os.path.join(download_dir, f"{img['project_id']}.png"),"wb")
                    response = requests.get(img['image_url'], headers=header)
                    file.write(response.content)
                    file.close()

        except Exception as e:
            logger.error(self.download_to.__qualname__, f"Error downloading {dataset} dataset: {e}")
            return False
        
        return True

    def dataset(self, dataset : str, download : bool, download_dir : str) -> DataFrame:
        data, header = self.access(dataset)
        imgs_ = {
                'dataset_name': [],
                'target': [],
                'image_url': [],
                'project_id': [],
                'image_path': [],
                'insertion_date': [],
                'metadata': [],
                'date_acquisition': [],
                'number_reports': [],
                }
        n_imgs = 0
        for img in data:
            image_path = os.path.join(download_dir, f"{img['project_id']}.png'")
            if img['dataset_name'] == 'imageamento_anonimizado_valid' or img['dataset_name'] == 'imageamento' or img['dataset_name'] == 'complete_imageamento_anonimizado_valid':
                imgs_['target'].append(0)
            else:
                imgs_['target'].append(int(img['metadata']['has_tb']))
            imgs_['dataset_name'].append(img['dataset_name'])
            imgs_['image_url'].append(img['image_url'])
            imgs_['project_id'].append(img['project_id'])
            imgs_['image_path'].append(image_path)
            imgs_['insertion_date'].append(img['insertion_date'])
            imgs_['metadata'].append(img['metadata'])
            imgs_['date_acquisition'].append(img['date_acquisition'])
            imgs_['number_reports'].append(img['number_reports'])
            n_imgs += 1
        df_data = DataFrame.from_dict(imgs_)
        df_data = df_data.sort_values('project_id')
        if download:
            if self.download_to(download_dir, dataset, data, df_data, header):
                logger.info(self.dataset.__qualname__, f'downloading {dataset} to {download_dir} OK!')
            else:
                logger.error(self.dataset.__qualname__, f'downloading {dataset} to {download_dir} FAILED!')
        return df_data