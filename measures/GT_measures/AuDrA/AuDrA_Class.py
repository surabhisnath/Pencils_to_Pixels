"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022, December 2). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity. https://doi.org/10.31234/osf.io/t63dm

"""
from __future__ import print_function
from argparse import ArgumentParser
from collections import namedtuple, OrderedDict
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models


class AuDrA(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.args = args
        self.cur_filename = "AuDrA_predictions.csv"

        # INITIALIZE MODEL
        try:
            self.model = models.__dict__[self.args.architecture](pretrained=self.args.pretrained)
        except:
            raise Exception('invalid architecture specified')

        # REMOVE LAST FULLY CONNECTED LAYER
        last_layer = list(self.model.children())[-1]
        if isinstance(last_layer, nn.Sequential):
            count = 0
            for layer in last_layer:
                if isinstance(layer, nn.Linear):
                    # fetch the first of the many Linear layers
                    count += 1
                    in_features = layer.in_features
                if count == 1:
                    break
        elif isinstance(last_layer, nn.Linear):
              in_features = last_layer.in_features

        # DEFINE NEW REGRESSION HEAD
        classifier = nn.Sequential(OrderedDict([
            ('bc1', nn.BatchNorm1d(in_features)),
            ('relu1', nn.ReLU()),
            ('fc1', nn.Linear(in_features, self.args.num_outputs, bias=True)),
        ]))

        # REPLACE THE CLASSIFIER WITH THE NEW REGRESSION HEAD
        if self.model.__dict__['_modules'].get('fc', None):
            self.model.fc = classifier
        else:
            self.model.classifier = classifier




    def forward(self, x):
        return self.model(x)

    def loss(self, pred_rating, true_rating):
        loss = self.args.loss_func(pred_rating, true_rating.view(-1,1))
        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'train_loss': pred_loss}

        # return{'loss': pred_loss, 'log': log, 'train_preds': pred, 'train_ratings': ratings.unsqueeze(dim=1), 'train_img_sums': sums, 'train_mae': mae}
        return{'loss': pred_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, labels, ratings = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        # print('\n\nPRED SHAPE')
        # print(pred.shape)
        log = {'val_loss': pred_loss, 'val_mae': mae, 'val_preds': pred, 'val_ratings': ratings.unsqueeze(dim=1), 'val_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()
        # maes = torch.cat([x['log']['val_mae'] for x in outputs])
        # print('\n LOSS SHAPE (STACK)')
        # print(val_loss_epoch.shape)
        # print('RATINGS SHAPE')
        # print(outputs[-1]['log']['ratings'].shape)
        # print(outputs[-1]['log']['ratings'])
        # print(outputs)
        # print(len(outputs))
        # print(outputs[0])

        ratings = torch.cat([x['log']['val_ratings'] for x in outputs])
        # print(ratings)
        # print('/n RATINGS MAXIMUM')
        # print(torch.min(ratings))
        # print('/n RATINGS MINIMUM')
        # print(torch.max(ratings))
        # print('/n RATINGS RANGE')
        # print(torch.max(ratings)-torch.min(ratings))
        # print('\n RATINGS SHAPE (CAT)')
        # print(ratings.shape)
        preds = torch.cat([x['log']['val_preds'] for x in outputs])
        # print('\n PREDS SHAPE (CAT)')
        # print(preds.shape)
        # print('\nRATINGS & shape')
        # print(ratings)
        # print(ratings.shape)
        # print('\nPREDS & shape')
        # print(preds)
        # print(preds.shape)
        # maes = torch.cat([x['log']['val_mae'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['val_img_sums'] for x in outputs]) # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1) # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        # print('\n SUMS SHAPE (CAT)')
        # print(sums.shape)
        # print('\n SUMS SHAPE (CAT) UNSQUEEZY')
        # print(sums.unsq.shape)
        # print('\n SUMS (CAT)')
        # print(sums)
        # print('\n SUMS (CAT) UNSQUEEZY')
        # print(sums.unsq)
        # print('\n SUMS SHAPE (STACK)')
        # print(sums.stack.shape)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv('validation_dataframe.csv')

        log = {'val_loss_epoch': val_loss_epoch,
               'val_correlation': correlation,
               'val_ink_correlation': ink_correlation,
               # 'val_ratings': ratings,
               # 'val_predictions': preds,
               # 'val_maes': maes,
               }

        return {'log': log, 'val_loss_epoch': val_loss_epoch}


    def test_step(self, batch, batch_idx):
        images, labels, ratings = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        unstandardized_imgs = [(img * self.args.img_stds + self.args.img_means) for img in images]
        sums = torch.stack([torch.sum(img) for img in unstandardized_imgs])

        log = {'test_loss': pred_loss, 'test_mae': mae, 'test_preds': pred, 'test_ratings': ratings.unsqueeze(dim=1), 'test_img_sums': sums}

        return{'loss': pred_loss, 'log': log}


    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()

        ratings = torch.cat([x['log']['test_ratings'] for x in outputs])
        preds = torch.cat([x['log']['test_preds'] for x in outputs])
        # maes = torch.cat([x['log']['test_mae'] for x in outputs])
        vx = ratings - torch.mean(ratings)
        vy = preds - torch.mean(preds)
        correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        sums = torch.cat([x['log']['test_img_sums'] for x in outputs]) # THIS IS THE ORIGINAL V WITHOUT UNSQUEEZE
        sums = sums.unsqueeze(1) # NEW MODIFICATION TO GET SHAPE TO (1107,1) INSTEAD OF (1107)
        vx = ratings - torch.mean(ratings)
        vy = sums - torch.mean(sums)
        ink_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        ratings_np = ratings.cpu()
        ratings_np = ratings_np.numpy()
        ratings_np = ratings_np.tolist()
        preds_np = preds.cpu()
        preds_np = preds_np.numpy()
        preds_np = preds_np.tolist()
        # maes_np = maes.numpy()
        df = pd.DataFrame({'ratings': ratings_np, 'predictions': preds_np})
        df.to_csv(self.cur_filename)

        log = {'test_loss_epoch': test_loss_epoch,
               'test_correlation': correlation,
               'test_ink_correlation': ink_correlation,
               # 'test_ratings': ratings,
               # 'test_predictions': preds,
               # 'test_maes': maes,
               }

        return {'log': log, 'test_loss_epoch': test_loss_epoch}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.args.learning_rate)
