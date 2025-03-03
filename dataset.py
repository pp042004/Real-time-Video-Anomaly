import numpy as np
import torch
from torch import Tensor
import os
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
import random



def train_loader_func_XD_Violence(modality, task='binary', Aug='0',batchsize=16):
    length = 32
    if task == 'binary':
        if modality == 'RGB' or modality == 'Flow':
            Base_path='/home/nit/Desktop/Gireesh/XD_Violance/{}'.format(modality)
            train_file_names = os.listdir(Base_path)
            # test_file_names = os.listdir(Base_path+'Test')
            train_x = []
            train_y = []
            for each in train_file_names:
                each = each[:-5]+Aug+'.npy'
                data = np.load(Base_path+'/'+each)
                # print(data.shape)
                if data.shape[0] == 0:
                    continue         
                new_f = np.zeros((length, data.shape[0])).astype(np.float32) #32x1024 
                r = np.linspace(0, data.shape[0]-1, length, dtype=int)
                # print(r)
                data = data[r]
                train_x.append(data)
                label = 0 if each.split('label_')[1][0] == 'A' else 1
                # print(data.shape, label)
                train_y.append(label)
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
        elif modality == 'both':
            train_data_x = []
            train_data_y = []
            for each_modality in ['RGB', 'Flow']:
                Base_path='/home/nit/Desktop/Gireesh/XD_Violance/{}'.format(each_modality)
                train_file_names = os.listdir(Base_path)
                train_x = []
                train_y = []
                for each in train_file_names:
                    each = each[:-5]+Aug+'.npy'
                    data = np.load(Base_path+'/'+each)
                    # print(data.shape)
                    if data.shape[0] == 0:
                        continue         
                    new_f = np.zeros((length, data.shape[0])).astype(np.float32) #32x1024 
                    r = np.linspace(0, data.shape[0]-1, length, dtype=int)
                    # print(r)
                    data = data[r]
                    train_x.append(data)
                    label = 0 if each.split('label_')[1][0] == 'A' else 1
                    # print(data.shape, label)
                    train_y.append(label)
                train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
                train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
                train_data_x.append(train_x_tensor)
            train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
        else:
            assert 1==1, print("Invalid input modality")
    elif task == 'multi':
        if modality == 'RGB' or modality == 'Flow':
            Base_path='/home/nit/Desktop/Gireesh/XD_Violance/{}'.format(modality)
            classes_dict = {'B1':0, 'B2':1, 'B4':2, 'B5':3, 'B6':4, 'G':5}
            train_file_names = os.listdir(Base_path)
            # test_file_names = os.listdir(Base_path+'Test')
            train_x = []
            train_y = []
            for each in train_file_names:
                if each.split('label_')[1][0] == 'A':
                    continue                   
                if len(each.split('label_')[1].split('-')) != 1 and each.split('label_')[1].split('-')[1] != '0':
                    continue
                each = each[:-5]+Aug+'.npy'
                data = np.load(Base_path+'/'+each)
                # print(data.shape)
                if data.shape[0] == 0:
                    continue         
                new_f = np.zeros((length, data.shape[0])).astype(np.float32) #32x1024 
                r = np.linspace(0, data.shape[0]-1, length, dtype=int)
                # print(r)
                data = data[r]
                train_x.append(data)
                if each.split('label_')[1][0] == 'G':
                    label = classes_dict['G']
                else:
                    label = classes_dict[each.split('label_')[1][:2]]
                # print(data.shape, each, label)
                train_y.append(label)
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.long).view(-1, 1).to(device)
        elif modality == 'both':
            train_data_x = []
            train_data_y = []
            classes_dict = {'B1':0, 'B2':1, 'B4':2, 'B5':3, 'B6':4, 'G':5}
            for each_modality in ['RGB', 'Flow']:
                Base_path='/home/nit/Desktop/Gireesh/XD_Violance/{}'.format(each_modality)
                train_file_names = os.listdir(Base_path)
                train_x = []
                train_y = []
                for each in train_file_names:
                    if each.split('label_')[1][0] == 'A':
                        continue                   
                    if len(each.split('label_')[1].split('-')) != 1 and each.split('label_')[1].split('-')[1] != '0':
                        continue
                    each = each[:-5]+Aug+'.npy'
                    data = np.load(Base_path+'/'+each)
                    # print(data.shape)
                    if data.shape[0] == 0:
                        continue         
                    new_f = np.zeros((length, data.shape[0])).astype(np.float32) #32x1024 
                    r = np.linspace(0, data.shape[0]-1, length, dtype=int)
                    # print(r)
                    data = data[r]
                    train_x.append(data)
                    if each.split('label_')[1][0] == 'G':
                        label = classes_dict['G']
                    else:
                        label = classes_dict[each.split('label_')[1][:2]]
                    # print(data.shape, each, label)
                    train_y.append(label)
                train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
                train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.long).view(-1, 1).to(device)
                train_data_x.append(train_x_tensor)
            train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
        else:
            assert 1==1, print("Invalid input modality")
    else:
        print("Invalid task is defined")
    # Define Dataset and DataLoader for batching
    batch_size = batchsize
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
           

def test_loader_func_XD_Violence(modality, task='binary', Aug='0',batchsize=16):
    length = 32
    if task == 'binary':
        if modality == 'RGB' or modality == 'Flow':
            Base_path='/home/nit/Desktop/Gireesh/XD_Violance/{}Test'.format(modality)
            train_file_names = os.listdir(Base_path)
            # test_file_names = os.listdir(Base_path+'Test')
            train_x = []
            train_y = []
            for each in train_file_names:
                each = each[:-5]+Aug+'.npy'
                data = np.load(Base_path+'/'+each)
                # print(data.shape)
                if data.shape[0] == 0:
                    continue         
                new_f = np.zeros((length, data.shape[0])).astype(np.float32) #32x1024 
                r = np.linspace(0, data.shape[0]-1, length, dtype=int)
                # print(r)
                data = data[r]
                train_x.append(data)
                label = 0 if each.split('label_')[1][0] == 'A' else 1
                # print(data.shape, label)
                train_y.append(label)
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
        elif modality == 'both':
            train_data_x = []
            train_data_y = []
            for each_modality in ['RGB', 'Flow']:
                Base_path='/home/nit/Desktop/Gireesh/XD_Violance/{}Test'.format(each_modality)
                train_file_names = os.listdir(Base_path)
                train_x = []
                train_y = []
                for each in train_file_names:
                    each = each[:-5]+Aug+'.npy'
                    data = np.load(Base_path+'/'+each)
                    # print(data.shape)
                    if data.shape[0] == 0:
                        continue         
                    new_f = np.zeros((length, data.shape[0])).astype(np.float32) #32x1024 
                    r = np.linspace(0, data.shape[0]-1, length, dtype=int)
                    # print(r)
                    data = data[r]
                    train_x.append(data)
                    label = 0 if each.split('label_')[1][0] == 'A' else 1
                    # print(data.shape, label)
                    train_y.append(label)
                train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
                train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
                train_data_x.append(train_x_tensor)
            train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
        else:
            assert 1==1, print("Invalid input modality")
    elif task == 'multi':
        if modality == 'RGB' or modality == 'Flow':
            Base_path='/home/nit/Desktop/Gireesh/XD_Violance/{}Test'.format(modality)
            classes_dict = {'B1':0, 'B2':1, 'B4':2, 'B5':3, 'B6':4, 'G':5}
            train_file_names = os.listdir(Base_path)
            # test_file_names = os.listdir(Base_path+'Test')
            train_x = []
            train_y = []
            for each in train_file_names:
                if each.split('label_')[1][0] == 'A':
                    continue                   
                if len(each.split('label_')[1].split('-')) != 1 and each.split('label_')[1].split('-')[1] != '0':
                    continue
                each = each[:-5]+Aug+'.npy'
                data = np.load(Base_path+'/'+each)
                # print(data.shape)
                if data.shape[0] == 0:
                    continue         
                new_f = np.zeros((length, data.shape[0])).astype(np.float32) #32x1024 
                r = np.linspace(0, data.shape[0]-1, length, dtype=int)
                # print(r)
                data = data[r]
                train_x.append(data)
                if each.split('label_')[1][0] == 'G':
                    label = classes_dict['G']
                else:
                    label = classes_dict[each.split('label_')[1][:2]]
                # print(data.shape, each, label)
                train_y.append(label)
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.long).view(-1, 1).to(device)
        elif modality == 'both':
            train_data_x = []
            train_data_y = []
            classes_dict = {'B1':0, 'B2':1, 'B4':2, 'B5':3, 'B6':4, 'G':5}
            for each_modality in ['RGB', 'Flow']:
                Base_path='/home/nit/Desktop/Gireesh/XD_Violance/{}Test'.format(each_modality)
                train_file_names = os.listdir(Base_path)
                train_x = []
                train_y = []
                for each in train_file_names:
                    if each.split('label_')[1][0] == 'A':
                        continue                   
                    if len(each.split('label_')[1].split('-')) != 1 and each.split('label_')[1].split('-')[1] != '0':
                        continue
                    each = each[:-5]+Aug+'.npy'
                    data = np.load(Base_path+'/'+each)
                    # print(data.shape)
                    if data.shape[0] == 0:
                        continue         
                    new_f = np.zeros((length, data.shape[0])).astype(np.float32) #32x1024 
                    r = np.linspace(0, data.shape[0]-1, length, dtype=int)
                    # print(r)
                    data = data[r]
                    train_x.append(data)
                    if each.split('label_')[1][0] == 'G':
                        label = classes_dict['G']
                    else:
                        label = classes_dict[each.split('label_')[1][:2]]
                    # print(data.shape, each, label)
                    train_y.append(label)
                train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
                train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.long).view(-1, 1).to(device)
                train_data_x.append(train_x_tensor)
            train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
        else:
            assert 1==1, print("Invalid input modality")
    else:
        print("Invalid task is defined")
    # Define Dataset and DataLoader for batching
    batch_size = batchsize
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
            
# test_loader_func(modality='both', task='multi', Aug='0',batchsize=16)


def train_loader_func_Shanghai_Tech(modality, batchsize=16):
    """
    modality is either RGB(all_rgbs), or Flow(all_flows), or Both
    """
    if modality == 'all_rgbs' or modality == 'all_flows':
        base_path = f'/home/nit/Desktop/Anomaly_Detection/ShanghaiTech/{modality}/train_shuffle1'
        folder_names = os.listdir(base_path)  # List all folders in the Train directory
        train_data = []
        
        for folder in folder_names:
            folder_path = os.path.join(base_path, folder)  # Get the full path to the folder
            file_names = os.listdir(folder_path)  # List all files in the folder
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)  # Full path to the file
                train_data.append((file_path, folder))  # Folder name is used as the label
        
        # print(f"Number of training samples: {len(train_data)}")
        train_x = []
        train_y = []
        for each in train_data:
            data = np.load(each[0])
            if data.shape[0] == 0:
                continue
            train_x.append(data)
            label = 0 if each[1] == "normal" else 1
            train_y.append(label)
        
        # Convert data to tensors

        train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
        train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
    elif modality == 'both':
        train_data_x = []
        train_data_y = []
        for each_modality in ['all_rgbs', 'all_flows']:
            base_path = f'/home/nit/Desktop/Anomaly_Detection/ShanghaiTech/{each_modality}/train_shuffle1'
            folder_names = os.listdir(base_path)  # List all folders in the Train directory
            train_data = []
            
            for folder in folder_names:
                folder_path = os.path.join(base_path, folder)  # Get the full path to the folder
                file_names = os.listdir(folder_path)  # List all files in the folder
                for file_name in file_names:
                    file_path = os.path.join(folder_path, file_name)  # Full path to the file
                    train_data.append((file_path, folder))  # Folder name is used as the label
            
            # print(f"Number of training samples: {len(train_data)}")
            train_x = []
            train_y = []
            for each in train_data:
                data = np.load(each[0])
                if data.shape[0] == 0:
                    continue
                train_x.append(data)
                label = 0 if each[1] == "normal" else 1
                train_y.append(label)
            
            # Convert data to tensors
        
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
            train_data_x.append(train_x_tensor)
        train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
        # print(train_x_tensor.shape, train_y_tensor.shape)
    else:
        assert 1==1, print("Invalid input modality")
        
    # Define Dataset and DataLoader for batching
    batch_size = batchsize
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
    # train_loader = train_loader_func('all_rgbs', 16)

def test_loader_func_Shanghai_Tech(modality, batchsize=16):
    """
    modality is either RGB(all_rgbs), or Flow(all_flows), or Both
    """
    if modality == 'all_rgbs' or modality == 'all_flows':
        base_path = f'/home/nit/Desktop/Anomaly_Detection/ShanghaiTech/{modality}/test_shuffle1'
        folder_names = os.listdir(base_path)  # List all folders in the Train directory
        train_data = []
        
        for folder in folder_names:
            folder_path = os.path.join(base_path, folder)  # Get the full path to the folder
            file_names = os.listdir(folder_path)  # List all files in the folder
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)  # Full path to the file
                train_data.append((file_path, folder))  # Folder name is used as the label
        
        # print(f"Number of training samples: {len(train_data)}")
        train_x = []
        train_y = []
        for each in train_data:
            data = np.load(each[0])
            if data.shape[0] == 0:
                continue
            train_x.append(data)
            label = 0 if each[1] == "normal" else 1
            train_y.append(label)
        # Convert data to tensors
        train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
        train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
    elif modality == 'both':
        train_data_x = []
        train_data_y = []
        for each_modality in ['all_rgbs', 'all_flows']:
            base_path = f'/home/nit/Desktop/Anomaly_Detection/ShanghaiTech/{each_modality}/test_shuffle1'
            folder_names = os.listdir(base_path)  # List all folders in the Train directory
            train_data = []
            
            for folder in folder_names:
                folder_path = os.path.join(base_path, folder)  # Get the full path to the folder
                file_names = os.listdir(folder_path)  # List all files in the folder
                for file_name in file_names:
                    file_path = os.path.join(folder_path, file_name)  # Full path to the file
                    train_data.append((file_path, folder))  # Folder name is used as the label
            
            # print(f"Number of training samples: {len(train_data)}")
            train_x = []
            train_y = []
            for each in train_data:
                data = np.load(each[0])
                if data.shape[0] == 0:
                    continue
                train_x.append(data)
                label = 0 if each[1] == "normal" else 1
                train_y.append(label)
            
            # Convert data to tensors
        
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
            train_data_x.append(train_x_tensor)
        train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
        # print(train_x_tensor.shape, train_y_tensor.shape)
    else:
        assert 1==1, print("Invalid input modality")
        
        # Define Dataset and DataLoader for batching
    batch_size = batchsize
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return test_loader
# test_loader = test_loader_func('all_flows')

def train_loader_func_UCF_Crime(modality, mode='multi', batchsize=16):
    """
    modality is either RGB(all_rgbs), or Flow(all_flows), or Both
    """
    if mode == 'binary':
        if modality == 'all_rgbs' or modality == 'all_flows':
            base_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/{modality}/'
            file_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/Anomaly_Train.txt'
            file_names = open(file_path).read().split()
            train_x = []
            train_y = []
            for each in file_names:
                data = np.load(base_path+each+'.npy')
                if data.shape[0] == 0:
                    continue
                train_x.append(data)
                label = 0 if "Normal_Videos_event" in each else 1
                train_y.append(label)
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
        elif modality == 'both':
            train_data_x = []
            train_data_y = []
            for each_modality in ['all_rgbs', 'all_flows']:
                base_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/{each_modality}/'
                file_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/Anomaly_Train.txt'
                file_names = open(file_path).read().split()
                train_x = []
                train_y = []
                for each in file_names:
                    data = np.load(base_path+each+'.npy')
                    if data.shape[0] == 0:
                        continue
                    train_x.append(data)
                    label = 0 if "Normal_Videos_event" in each else 1
                    train_y.append(label)
                train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
                train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
                # print(train_x_tensor.shape, train_y_tensor.shape)
                train_data_x.append(train_x_tensor)
            train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
            # print(train_x_tensor.shape, train_y_tensor.shape)
        else:
            assert 1==1, print("Invalid input modality")
    # for multi-class classification
    if mode == 'multi':
        fid = random.randint(1, 4)
        # fid=1
        anomalies = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
        if modality == 'all_rgbs' or modality == 'all_flows':
            base_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/{modality}/'
            file_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/splits/train_00{fid}.txt'
            file_names = open(file_path).read().split()
            train_x = []
            train_y = []
            for each in file_names:
                data = np.load(base_path+each+'.npy')
                if data.shape[0] == 0:
                    continue
                class_name = each.split('/')[0]
                if 'Normal' in class_name:
                    continue
                # data = data/np.linalg.norm(data, ord=2, axis=-1, keepdims=True) 
                train_x.append(data)
                # class_name = each.split('/')[0]
                class_id = anomalies.index(class_name)
                label = class_id
                train_y.append(label)
                # print(each, class_name, class_id)
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.long).view(-1, 1).to(device)
        if modality == 'both':
            train_data_x = []
            train_data_y = []
            for each_modality in ['all_rgbs', 'all_flows']:
                base_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/{each_modality}/'
                file_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/splits/train_00{fid}.txt'
                file_names = open(file_path).read().split()
                train_x = []
                train_y = []
                for each in file_names:
                    data = np.load(base_path+each+'.npy')
                    if data.shape[0] == 0:
                        continue
                    class_name = each.split('/')[0]
                    if 'Normal' in class_name:
                        continue
                    # data = data/np.linalg.norm(data, ord=2, axis=-1, keepdims=True)
                    train_x.append(data)
                    # class_name = each.split('/')[0]
                    class_id = anomalies.index(class_name)
                    label = class_id
                    train_y.append(label)
                train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
                train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.long).view(-1, 1).to(device)
                train_data_x.append(train_x_tensor)
            train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
    # Define Dataset and DataLoader for batching
    batch_size = batchsize
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
# train_loader = train_loader_func('all_rgbs', 16)

def test_loader_func_UCF_Crime(modality, mode='multi', batchsize=16):
    """
    modality is either RGB(all_rgbs), or Flow(all_flows), or Both
    """
    if mode == 'binary':
        if modality == 'all_rgbs' or modality == 'all_flows':
            base_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/{modality}/'
            file_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/Anomaly_Test.txt'
            file_names = open(file_path).read().split()
            train_x = []
            train_y = []
            for each in file_names:
                data = np.load(base_path+each+'.npy')
                if data.shape[0] == 0:
                    continue
                train_x.append(data)
                label = 0 if "Normal_Videos_event" in each else 1
                train_y.append(label)
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
        elif modality == 'both':
            train_data_x = []
            train_data_y = []
            for each_modality in ['all_rgbs', 'all_flows']:
                base_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/{each_modality}/'
                file_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/Anomaly_Test.txt'
                file_names = open(file_path).read().split()
                train_x = []
                train_y = []
                for each in file_names:
                    data = np.load(base_path+each+'.npy')
                    if data.shape[0] == 0:
                        continue
                    train_x.append(data)
                    label = 0 if "Normal_Videos_event" in each else 1
                    train_y.append(label)
                train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
                train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.float32).view(-1, 1).to(device)
                # print(train_x_tensor.shape, train_y_tensor.shape)
                train_data_x.append(train_x_tensor)
            train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
            # print(train_x_tensor.shape, train_y_tensor.shape)
        else:
            assert 1==1, print("Invalid input modality")
    #for multi-class classification
    if mode == 'multi':
        fid = random.randint(1, 4)
        # fid=3
        anomalies = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
        if modality == 'all_rgbs' or modality == 'all_flows':
            base_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/{modality}/'
            file_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/splits/test_00{fid}.txt'
            file_names = open(file_path).read().split()
            train_x = []
            train_y = []
            for each in file_names:
                data = np.load(base_path+each+'.npy')
                if data.shape[0] == 0:
                    continue
                class_name = each.split('/')[0]
                if 'Normal' in class_name:
                    continue
                # data = data/np.linalg.norm(data, ord=2, axis=-1, keepdims=True)
                train_x.append(data)
                # class_name = each.split('/')[0]
                class_id = anomalies.index(class_name)
                label = class_id
                train_y.append(label)
                # print(each, class_name, class_id)
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
            train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.long).view(-1, 1).to(device)
        if modality == 'both':
            train_data_x = []
            for each_modality in ['all_rgbs', 'all_flows']:
                base_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/{each_modality}/'
                file_path = f'/home/nit/Desktop/Gireesh/UCF_Crime/UCF-Crime/splits/test_00{fid}.txt'
                file_names = open(file_path).read().split()
                train_x = []
                train_y = []
                for each in file_names:
                    data = np.load(base_path+each+'.npy')
                    if data.shape[0] == 0:
                        continue
                    class_name = each.split('/')[0]
                    if 'Normal' in class_name:
                        continue
                    # data = data/np.linalg.norm(data, ord=2, axis=-1, keepdims=True)
                    train_x.append(data)
                    class_id = anomalies.index(class_name)
                    label = class_id
                    train_y.append(label)
                train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
                train_y_tensor = torch.tensor(np.array(train_y), dtype=torch.long).view(-1, 1).to(device)
                train_data_x.append(train_x_tensor)
            train_x_tensor = torch.cat((train_data_x[0], train_data_x[1]), dim=2)
    # Define Dataset and DataLoader for batching
    batch_size = batchsize
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return test_loader
# test_loader = test_loader_func('all_flows')