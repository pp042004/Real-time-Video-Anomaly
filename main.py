import os
import re
import torch
import math
from torch import Tensor
from tqdm import tqdm
from dataset import *
from train import *
from time import time, gmtime, strftime
from model import ODEVAE
from test import test_multi, test
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_std = 0.02
warmup_epochs = 60
batchsize = 128
max_len = 32 #maximum length of the features
n_epochs = 1000
Aug = '0'
######################################
"""
Arguments to control the entire program
"""
dataset_name = 'UCF_Crime' #XD_Volence or Shanghai_Tech or UCF_Crime
task = 'multi' # task value either binary or multi
n_classes = 1 #for binary classification
classifier_name = 'BERT' #BERT or LSTM
train_mode = False #for training it is True otherwise it is False
if train_mode:
    test_mode = False
else:
    test_mode = True
exp_name = "multi_Class_Transfomer"
#######################################

if dataset_name == 'XD_Violence':
    modalities = ['RGB', 'Flow', 'both']
elif dataset_name == 'UCF_Crime' or dataset_name == 'ShanghaiTech':
    modalities = [ 'all_rgbs', 'all_flows', 'both']
    # modalities = ['all_rgbs']
else:
    assert 1==1, print("Choose a Proper dataset")

Exp_name = "<=======>LatentNODE with Transformer as Decoder for {} classification Task on {} dataset<=======>".format(task, dataset_name)
curr = time()
fp = open('results_log_{}.txt'.format(dataset_name), 'a')
fp.write("\n"+Exp_name+str(strftime("%a, %d %b %Y %H:%M:%S", gmtime(curr)))+"\n")

all_results = dict() 

for modality in modalities:
    if dataset_name == 'XD_Violence':
        train_loader = train_loader_func_XD_Violence(modality, task, Aug='0',batchsize=16)
        test_loader = test_loader_func_XD_Violence(modality, task, Aug='0',batchsize=16)
        if task == 'multi':
            n_classes = 6
    if dataset_name == 'UCF_Crime':
        train_loader = train_loader_func_UCF_Crime(modality, task, batchsize)
        test_loader = test_loader_func_UCF_Crime(modality, task, batchsize)
        if task == 'multi':
            n_classes = 13
    if dataset_name == 'ShanghaiTech':
        train_loader = train_loader_func_Shanghai_Tech(modality, batchsize)
        test_loader = test_loader_func_Shanghai_Tech(modality, batchsize)

    observations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # observations = [0.1, 0.3, 0.5, 0.7, 0.9]
    for observation in observations:
        path_to_save_model = f'/home/nit/Desktop/Gireesh/{dataset_name}_Exp/{modality}/{exp_name}/latent_NODE_{observation*100}.pth'
        if not os.path.isdir(f'/home/nit/Desktop/Gireesh/{dataset_name}_Exp/{modality}/{exp_name}'):
            os.mkdir(f'/home/nit/Desktop/Gireesh/{dataset_name}_Exp/{modality}/{exp_name}')
            print("directory created")
        if train_mode:
            print(f"******************* New percentage of observation started with {observation} in {modality}*****************")
            train(train_loader, test_loader, modality, classifier_name, n_classes, observation, task, n_epochs, path_to_save_model, dataset_name)    
        
        elif test_mode:
            print(f"***New percentage of observation started with {observation} in {modality} task performed {task} on {dataset_name}*****************")
            if modality == 'all_rgbs' or modality == 'all_flows' or modality == 'RGB' or modality == 'Flow':
                model = ODEVAE(1024, 512, 256, n_classes, 33-int(max_len*observation), classifier_name)
            elif modality == 'both':
                model = ODEVAE(2048, 512, 256, n_classes,33-int(max_len*observation), classifier_name)
            else:
                assert 1==1, print("Invalid Input modality")
            model = model.to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Number of parameters:", n_params)
            if task == 'binary':
                model.load_state_dict(torch.load(path_to_save_model, weights_only=True))
                model.eval()
                auc, ap = test(test_loader, observation, model)
                print(f'Video_AUC:{round(auc*100, 2)} Video_AP:{round(ap*100, 2)}')
            if task == 'multi':
                print(path_to_save_model)
                # print(model)
                model.load_state_dict(torch.load(path_to_save_model, weights_only=True))
                model.eval()
                top1_acc, top5_acc = test_multi(test_loader, observation, model)
                # print(f'Video_AUC:{round(auc*100, 2)} Video_AP:{round(ap*100, 2)} top1_acc:{round(top1_acc*100, 2)} top5_acc:{round(top5_acc*100, 2)}')
                print(f'Video_AUC:{top1_acc} Video_AP:{top5_acc}')
fp.close()