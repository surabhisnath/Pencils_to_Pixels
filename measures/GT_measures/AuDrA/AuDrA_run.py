"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022, December 2). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity. https://doi.org/10.31234/osf.io/t63dm

"""
from argparse import ArgumentParser
from AuDrA_DataModule import user_Dataloader
from AuDrA_Class import AuDrA
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn

creators = ["children", "adults", "AI/prompt_1", "AI/prompt_2", "AI/prompt_3"]

for creator in creators:
    for stim in ["G", "I", "R"]:
        print(creator, stim)
        data_dir = f"../../../data/{creator}/stimuli_{stim}/"

        #  MODEL CONFIGURATION (best-fitting settings)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Running AuDrA on","GPU" if device.type == "cuda" else "CPU")
        parser = ArgumentParser()
        parser.add_argument('--architecture', default = 'resnet18')
        parser.add_argument('--pretrained', default = True)
        parser.add_argument('--in_shape', default = [3,224,224], type = int)
        parser.add_argument('--img_means', default = 0.1612 , type = float)
        parser.add_argument('--img_stds', default = 0.4075, type = float)
        parser.add_argument('--num_outputs', default = 1, type = int)
        parser.add_argument('--learning_rate', default = 0.00034664640432471026)
        parser.add_argument('--batch_size', default = 16)
        parser.add_argument('--train_pct', default = 0.7, type = float)
        parser.add_argument('--val_pct', default = 0.1, type = float)
        parser.add_argument('--loss_func', default = nn.MSELoss(), type = object)
        parser.add_argument('--num_workers', default = 1)
        parser.add_argument('--data_dir', default = data_dir)
        args = parser.parse_args()

        #  LOAD TRAINED AUDRA MODEL
        model = AuDrA(args)
        model_weights = torch.load('AuDrA_trained.ckpt', map_location = device)
        model.load_state_dict(model_weights['state_dict'])
        model.eval()
        model.to(device)

        #  LOAD IMAGES FOR PREDICTIONS
        dataloader = user_Dataloader(args = args)

        #  GET AuDrA PREDICTIONS
        filenames = [] #storage for image names
        predictions = [] #AuDrA predictions (in JRT theta values)
        for _, img in enumerate(dataloader):
            fname = img[0][0]
            x = img[1]
            x= x.to(device)
            pred = model.forward(x).item()
            filenames.append(fname)
            predictions.append(pred)
            print(fname,": ",str(pred))

        #  SAVE OUTPUT FILE
        out_df = pd.DataFrame(zip(filenames, predictions), columns= ["filenames", "predictions"])
        out_df.to_csv(f"output/{creator.replace('_', '').replace('/', '_')}_{stim}.csv", index=False)