# from __future__ import annotations
import os
import random

import numpy
import torch
from pandas import DataFrame, concat
from torch.utils.data import DataLoader

from slasher3.device import get_device
from slasher3.log import logger
from slasher3.nets.net import Net
from slasher3.nets.slashernet import SlasherNet
from slasher3.xray import XRays


def sp(a, b):
    return ((a + b)/2.0 * (a*b)**(1/2.0))**(1/2.0)

# CHILDREN WILL INITIATE SOLVER PARAMETERS AND RUN SOLVE
class Solver:
    name : str
    override_device : bool = False
    device = None
    random_seed : int = 0
    database : XRays
    sampler_kind : str
    max_images : int
    model_class : Net
    batch_size : int
    export_path : str
    epochs : int
    training_lr : float
    patience : int
    output_width : int
    solver_ready : bool = False

    def __init__(self, name, override_device, random_seed, database, max_images, batch_size, export_path, epochs, training_lr, patience, output_width):
        self.name = name
        self.override_device = override_device
        self.device = get_device(self.override_device)
        self.random_seed = random_seed
        self.database = database
        self.sampler_kind = None # Child should initialize
        self.max_images = max_images
        self.batch_size = batch_size
        self.export_path = export_path
        self.epochs = epochs
        self.training_lr = training_lr
        self.patience = patience
        self.output_width = output_width
        self.solver_ready = self.set_solver()

    def set_solver(self) -> bool:
        pass

    def solve(self) -> tuple[Net, DataFrame]: # MODEL / RESULTS
        if self.solver_ready:
            torch.manual_seed(self.random_seed)
            random.seed(self.random_seed)
            numpy.random.seed(self.random_seed)
            logger.info(self.solve.__qualname__, f"breeding {self.name}...")
            result_list = []
            for fold in self.database.folds:
                # for test in self.database.test: # UNCOMMENT AND TAB BELOW FOR FULL SEIXAS LEARNING
                test = fold + 1
                logger.info(self.solve.__qualname__, f"Starting training for sort {fold} and test {test}")
                model_name = f"{self.name.upper()}-K{fold}T{test}"
                train_db = self.database.sample(self.sampler_kind, fold, test, "train", "real", self.max_images, shuffle=True)
                validation_db = self.database.sample(self.sampler_kind, fold, test, "val", "real", self.max_images, shuffle=True)
                training_loader = DataLoader(train_db, batch_size=self.batch_size, shuffle=True)
                validation_loader = DataLoader(validation_db, batch_size=self.batch_size, shuffle=True)
                model = self.model_class(model_name, self.export_path, self.device, self.output_width, training_loader, validation_loader, self.epochs, self.training_lr, self.patience)
                results = model.results
                results['model_name'] = model_name
                results['fold'] = fold
                results['model'] = model
                result_list.append(results)
                # UNCOMMENT FOR FASTER DEBUG TRAINING :)
                # if test > 2:
                #     break

            training_results = concat(result_list, ignore_index=True)

            # GENERATE df_y_vs_y
            training_results['fold'] = [x[-1] for x in training_results.name.str.split(pat='-')]
            training_results['ix_train'] = training_results.fold.str[0:2]
            training_results['ix_test'] = training_results.fold.str[2:]

            # FIND BEST FROM FOLD
            ix_best = training_results[training_results.is_best].groupby(["name"])["epoch"].idxmax().to_list()
            results = training_results.iloc[ix_best].copy()

            # FINDING BEST MODEL
            best_fold_data = results.sort_values(by="validation_mean_accuracy", ascending=False).iloc[0]
            best_model = best_fold_data.model

            # model.load_state_dict(torch.load(model_pickle_name))
            # model.to(self.device)
            # model.eval()
            # fold_best = int(best_model_data["ix_train"][-1])
            # test_best = int(best_model_data["ix_test"][-1])

            # TEST Dataset
            # test_db = self.database.sample(self.sampler_kind, fold_best, test_best, "test", "real", self.max_images, shuffle=True)
            # test_loader = DataLoader(test_db, batch_size=self.batch_size, shuffle=True)
            # n_batches = len(test_loader)
            # test_batch_list = []
            # with torch.no_grad():
            #     for ix_batch, (x_batch, y_batch) in enumerate(test_loader):
            #         x = x_batch.to(self.device, dtype=torch.float32)
            #         y = y_batch[:, None].to(self.device, dtype=torch.float32)
            #         y_est = model(x)
            #         ya = y.cpu().detach().numpy()
            #         ya_est = y_est.cpu().detach().numpy()
            #         for yi, y_est_i in zip(ya, ya_est):
            #             test_batch_list.append({"y" : yi[0], "y_est": y_est_i[0]})

            # Test CSV
            # y_results = DataFrame.from_dict(test_batch_list)
            # y_results_csv = os.path.join(self.export_path, f"{self.config_name}_y_results.csv")
            # y_results.to_csv(y_results_csv, sep=';', decimal=',', index=False)

            # SAVE MODEL
            best_model_path = os.path.join(self.export_path, f"{self.name}.net")
            torch.save(best_model.net.state_dict(), best_model_path)

            # SAVE CSVS
            training_results_csv = os.path.join(self.export_path, f"{self.name}_training_results.csv")
            training_results.to_csv(training_results_csv, sep=';', decimal=',', index=False)

            results_csv = os.path.join(self.export_path, f'{self.name}_results.csv')
            results.to_csv(results_csv, sep=';', decimal=",", index=False)

            # mean_results_csv = os.path.join(self.export_path, f'{self.config_name}_mean_results.csv')
            # mean_results.to_csv(mean_results_csv, sep=';', decimal=",", index=True)
            return best_model, results
        else:
            logger.warn(self.solve.__qualname__, f"{self.name} Solver not ready...")

class Slasher(Solver):
    def set_solver(self) -> bool:
        self.model_class = SlasherNet
        self.sampler_kind = "segment"
        return True