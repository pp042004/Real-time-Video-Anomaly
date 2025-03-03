import torch
import numpy as np
import torch.nn as nn
from model import *
from time import time, strftime, gmtime
from tqdm import tqdm
from test import *
from Early_Stop import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ablation_study = True
max_len = 32
noise_std = 0.02

def train(train_loader, test_loader, modality, classifier_name, n_classes, observation, task, n_epochs, path_to_save_model, dataset_name):
    print("Path to save model:", path_to_save_model)
    max_auc = 0
    fp = open('results_log_{}.txt'.format(dataset_name), 'a')
    if modality == 'all_rgbs' or modality == 'all_flows' or modality == 'RGB' or modality == 'Flow':
        model = ODEVAE(1024, 512, 256, n_classes, 33-int(max_len*observation), classifier_name)
    elif modality == 'both':
        model = ODEVAE(2048, 512, 256, n_classes,33-int(max_len*observation), classifier_name)
    else:
        assert 1==1, print("Invalid Input modality")

    optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.001)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", n_params)
    early_stopping = EarlyStopping(patience=100, min_delta=5.0)
    for epoch_idx in tqdm(range(1, n_epochs+1), ncols=100, desc="Training Steps"):
        # train_loader = train_loader_func_UCF_Crime(modality, task, 128)
        losses = []
        for i, (batch_sequence, batch_label) in enumerate(train_loader):
            optim.zero_grad()
            x = batch_sequence
            t = int(observation*x.shape[1])
            x_p, _, z_mean, z_log_var, lstm_out = model(x, t)
            x_p = x_p.permute(1, 0, 2)
            if not ablation_study:
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
                loss = 0.5 * ((x[:, t:, :]-x_p)**2).sum(1).sum(1) / noise_std**2 + kl_loss
            else: 
                loss = 0.5 * ((x[:, t:, :]-x_p)**2).sum(1).sum(1) / noise_std**2
            loss = torch.mean(loss)
            loss /= t
            if task == 'binary':
                classification_loss = nn.BCEWithLogitsLoss()(lstm_out, batch_label)
            elif task == 'multi':
                classification_loss = nn.CrossEntropyLoss()(lstm_out, batch_label.squeeze()) #for multi-class classification
            # if epoch_idx > warmup_epochs:
            loss = loss + classification_loss
            loss.backward()
            optim.step()
            losses.append(loss.item())
            cls_loss = classification_loss.item()
        if task == 'binary':
            auc, ap = test(test_loader, observation, model)
        elif task == 'multi':
            auc, ap = test_multi(test_loader, observation, model)
        if auc > max_auc:
            max_auc = auc
            if task == 'binary':
                fp.write(str(f'Video_AUC:{round(auc*100, 2)} Video_AP:{round(ap*100, 2)}'+"\n"))
            if task == 'multi':
                fp.write(str(f'top1_acc:{auc} top5_acc:{ap}'+"\n"))        
            torch.save(model.state_dict(), path_to_save_model)
        # print(f"Epoch [{epoch_idx+1}/{n_epochs}], Training Loss: {np.mean(losses), np.median(losses), cls_loss}, Validation Loss: {auc, ap}")
        # early_stopping(auc)
        # if early_stopping.early_stop:
        #     print("Early stopping triggered. Restoring best model weights.")
        #     print(f"Epoch [{epoch_idx+1}/{n_epochs}], Training Loss: {np.mean(losses), np.median(losses), cls_loss}, Scores AUC and AP: {auc*100, ap*100}")
        #     torch.save(model.state_dict(), path_to_save_model)
        #     if task == 'binary':
        #         fp.write(str(f'Video_AUC:{round(auc*100, 2)} Video_AP:{round(ap*100, 2)}'+"\n"))
        #     if task == 'multi':
        #         fp.write(str(f'top1_acc:{auc} top5_acc:{ap}'+"\n"))  
        #     break
        if epoch_idx%5 == 0:
            curr = time()
            if epoch_idx%100 == 0:
                print(str(strftime("%a, %d %b %Y %H:%M:%S", gmtime(curr))))
                print(f"Epoch {epoch_idx}/{n_epochs}")
                print(np.mean(losses), np.median(losses), cls_loss)
                if task == 'multi':
                    print(f'Video_top1_acc:{auc} Video_top5_acc:{ap}')
                if task == 'binary':
                    print(f'AUC:{round(auc*100, 2)} AP:{round(ap*100, 2)}')
            curr = time()
            fp.write(str(f"Epoch {epoch_idx}/{n_epochs}")+str(strftime("%a, %d %b %Y %H:%M:%S", gmtime(curr))))
            fp.write(str(f"Epoch {epoch_idx}/{n_epochs}"+"\n"+str(np.mean(losses), )+str(np.median(losses),)+str(cls_loss)+"\n"))
            # if task == 'binary':
            #     auc, ap = test(test_loader, observation, model)
            # elif task == 'multi':
            #     auc, ap = test_multi(test_loader, observation, model)
            # print("Model saved at:", path_to_save_model)
