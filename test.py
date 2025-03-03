from dataset import *
import torch
from sklearn import metrics
from time import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def test(test_loader, observed_per, model, feature_extraction=False):
    # test_loader = test_loader_func_Shanghai_Tech(modality, 16)
    loss = torch.tensor(0, dtype = float)
    predictions = []
    true_labels = []
    threshold = 0.5
    with torch.no_grad():
        t0 = time() 
        for i, (batch_sequence, batch_label) in enumerate(test_loader):
            x = batch_sequence
            x = x.to(device)
            t = int(observed_per*x.shape[1])
            x_p, z, z_mean, z_log_var,lstm_output  = model(x, t)
            x_p = x_p.permute(1, 0, 2)
            for each in lstm_output.tolist():
                for each1 in each:
                    predictions.append(each1)
            for each in batch_label.tolist():
                for each1 in each:
                    true_labels.append(each1)
       
    if feature_extraction:
        return predictions
    t1 = time()
    num_videos = len(true_labels)
    time_per_Video = (t1-t0)/(num_videos) 
    # fpv_video = 1/time_per_Video  
    # Calculate AUC score
    fpr_all, tpr_all, thresholds_all = metrics.roc_curve(true_labels, predictions, pos_label=1)
    vid_auc = metrics.auc(fpr_all, tpr_all)
    vid_ap = metrics.average_precision_score(true_labels, predictions, pos_label=1)
    # print(f'Video_AUC:{round(vid_auc*100, 2)} Video_AP:{round(vid_ap*100, 2)} time per Video = {time_per_Video}')
    return vid_auc, vid_ap

def test_multi(test_loader, observed_per, model,  feature_extraction=False):
    # test_loader = test_loader_func_UCF_Crime(modality, 'multi', 16)
    top1_acc = torch.tensor(0, dtype = float)
    top5_acc = torch.tensor(0, dtype = float)
    preds = []
    gt = []
    with torch.no_grad():
        t0 = time() 
        for i, (batch_sequence, batch_label) in enumerate(test_loader):
            x = batch_sequence
            t = int(observed_per*x.shape[1])
            x_p, z, z_mean, z_log_var,lstm_output  = model(x, t)
            acc1, acc5 = accuracy(lstm_output, batch_label, topk=(1,5))
            # print(len(acc1),len(acc5), acc1, acc5)
            top1_acc += acc1[0].to('cpu')
            top5_acc += acc5[0].to('cpu')
            for each in lstm_output.tolist():
                for each1 in each:
                    preds.append(each1)
            for each in batch_label.tolist():
                for each1 in each:
                    gt.append(each1)
    if feature_extraction:
        return preds
    t1 = time()
    num_videos = len(gt)
    time_per_Video = (t1-t0)/(num_videos) 
    top1_acc = round(top1_acc.item()/((i+1)), 2)
    top5_acc = round(top5_acc.item()/((i+1)), 2)
    # print("top1_Accuracy: ", top1_acc, "top5_Accuracy: ", top5_acc)
    return top1_acc, top5_acc
# test()