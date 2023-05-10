import sys
import time
import os

import numpy
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, jaccard_score)

from .net import Net
from .unet import Unet_Builder
from slasher3.log import logger
from slasher3.xray import XRayDataset


def calculate_jaccard_accuracy(y_estimate, y, threshold=0.5):
    """ Ground truth """
    y = y.cpu().detach().numpy()
    y = y > threshold
    y = y.astype(numpy.uint8)
    y = y.reshape(-1)

    """ Prediction """
    y_estimate = y_estimate.cpu().detach().numpy()
    y_estimate = y_estimate > threshold
    y_estimate = y_estimate.astype(numpy.uint8)
    y_estimate = y_estimate.reshape(-1)

    jaccard = jaccard_score(y, y_estimate)
    accuracy = accuracy_score(y, y_estimate)

    return jaccard, accuracy

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class SlasherNet(Net):
    def fit(self):
        self.net = Unet_Builder(input_channel=1, output_channel=1)
        self.net = self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.training_lr)
        self.loss_function = DiceBCELoss()
        self.score_function = calculate_jaccard_accuracy
        # """ fiting model to dataloaders, saving best weights and showing results """
        best_validation_loss = float("inf")
        results_list = []
        t_start = time.time()
        unimproved_epochs = 0
        for epoch in range(self.epochs):
            training_mean_loss, training_mean_jaccard, training_mean_accuracy = self.train_batches()
            validation_mean_loss, validation_mean_jaccard, validation_mean_accuracy = self.evaluate_batches()

            is_best = False
            if validation_mean_loss < best_validation_loss:
                is_best = True
                output_filename = f"{self.name}.net"
                output_path = os.path.join(self.net_folder, output_filename)
                is_best_str = '*** BEST ***'
                best_validation_loss = validation_mean_loss
                # torch.save(self.net.state_dict(), output_path)
                unimproved_epochs = 0
            else:
                is_best_str = ''
                unimproved_epochs += 1
                if unimproved_epochs > self.patience:
                    logger.info(SlasherNet.fit.__qualname__, f"{unimproved_epochs} unimproved epochs exceeded {self.patience} patience, stopping...")
                    break
            epoch_results_record = self.name, epoch+1, training_mean_loss,training_mean_accuracy,training_mean_jaccard, validation_mean_loss, validation_mean_accuracy, validation_mean_jaccard, is_best
            results_list.append(epoch_results_record)
            logger.info(SlasherNet.fit.__qualname__,f"{self.name} {epoch+1}/{self.epochs} Epochs, T/V mean loss {training_mean_loss:.3f}/{validation_mean_loss:.3f}, accuracy {100.0*training_mean_accuracy:.1f}%/{100.0*validation_mean_accuracy:.1f}%, jaccard {100.0*training_mean_jaccard:.1f}%/{100.0*validation_mean_jaccard:.1f}% {is_best_str}")

        period = time.time() - t_start
        logger.info(SlasherNet.fit.__qualname__,'Training complete in {:.0f}m {:.0f}s'.format(period // 60, period % 60))
        df_results = pandas.DataFrame.from_records(results_list)
        df_results.columns = ['name', 'epoch', 'training_mean_loss', 'training_mean_accuracy', 'training_mean_jaccard', 'validation_mean_loss', 'validation_mean_accuracy', 'validation_mean_jaccard', 'is_best']
        return df_results

    def train_batches(self):
        loss_cumsum = 0.0
        jaccard_cumsum = 0.0
        accuracy_cumsum = 0.0

        steps = len(self.training_loader)
        self.net.train()
        for i, (x, y) in enumerate(self.training_loader):
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            y_estimate = self.net(x)
            loss = self.loss_function(y_estimate, y)
            loss.backward()
            
            jaccard, accuracy = self.score_function(y_estimate, y)
            jaccard_cumsum += jaccard
            accuracy_cumsum += accuracy
            
            self.optimizer.step()
            loss_cumsum += loss.item()

        epoch_mean_loss = loss_cumsum/steps
        epoch_mean_jaccard = jaccard_cumsum/steps
        epoch_mean_accuracy = accuracy_cumsum/steps
        
        return epoch_mean_loss, epoch_mean_jaccard, epoch_mean_accuracy

    def evaluate_batches(self):
        loss_cumsum = 0.0
        jaccard_cumsum = 0.0
        accuracy_cumsum = 0.0
        steps = len(self.validation_loader)
        self.net.eval()
        with torch.no_grad():
            for x, y in self.validation_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)

                y_estimate = self.net(x)
                loss = self.loss_function(y_estimate, y)
                loss_cumsum += loss.item()
                jaccard, accuracy = self.score_function(y_estimate, y)
                jaccard_cumsum += jaccard
                accuracy_cumsum += accuracy

            mean_loss = loss_cumsum/steps
            mean_jaccard = jaccard_cumsum/steps
            mean_accuracy = accuracy_cumsum/steps
        
        return mean_loss, mean_jaccard, mean_accuracy
    
    def predict(self, to_mask : XRayDataset):
        self.net.eval()
        to_mask_loader = DataLoader(to_mask, shuffle=True)
        to_mask_batch_list = []
        with torch.no_grad():
            for x_batch, y_batch in to_mask_loader:
                x = x_batch.to(self.device, dtype=torch.float32)
                # y = y_batch[:, None].to(self.device, dtype=torch.float32)
                to_mask_batch_list.append((x.cpu().detach().numpy(), self.net(x).cpu().detach().numpy()))
        
        return pandas.DataFrame.from_records(to_mask_batch_list, columns=['image','mask'])
    
    def from_file(self, net_save: str):
        self.net = Unet_Builder(input_channel=1, output_channel=1)
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(net_save))
        self.net.eval()