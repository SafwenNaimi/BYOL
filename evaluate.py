# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:43:47 2021

@author: Safwen
"""

import utils
from tqdm import tqdm
from itertools import islice
import torch
import os

import models
import torch.nn as nn


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

torch.set_default_tensor_type(dtype)

#%%

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
    
def compute_acc(output,target,topk=(1,)):
    
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    return res
#    correct = (output == targets).float().sum()
#    num_samples=output.shape[0]
#    return correct/num_samples
def validate(epoch, model, device, dataloader, criterion, args, writer):
    """ Test loop, print metrics """
    progbar = tqdm(total=len(dataloader), desc='Val')

    
    loss_record = RunningAverage()
    acc_record = RunningAverage()
    model.eval()
    with torch.no_grad():
    #    for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
        for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
    
            # measure accuracy and record loss
            acc = compute_acc(output, label)
    #        acc_record.update(100 * acc[0].item())
            acc_record.update(100*acc[0].item()/data.size(0))
            loss_record.update(loss.item())
            #print('val Step: {}/{} Loss: {:.4f} \t Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))
            progbar.set_description('Val (loss=%.4f)' % (loss_record()))
            progbar.update(1)

    writer.add_scalar('validation/Loss_epoch', loss_record(), epoch)
    writer.add_scalar('validation/Acc_epoch', acc_record(), epoch)
    
    return loss_record(),acc_record()

def test( model, device, dataloader, criterion, args):
    """ Test loop, print metrics """
    loss_record = RunningAverage()
    acc_record = RunningAverage()
    model.eval()
    with torch.no_grad():
     #   for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
        for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
    
            # measure accuracy and record loss
            acc = compute_acc(output, label)
    #        acc_record.update(100 * acc[0].item())
            acc_record.update(100*acc[0].item()/data.size(0))
            loss_record.update(loss.item())
#            print('Test Step: {}/{} Loss: {:.4f} \t Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))

    return loss_record(),acc_record()
#%%
import yaml
if __name__=='__main__':
    class dotdict(dict):
   
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def load_yaml(config_file,config_type='dict'):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        #params = yaml.load(f,Loader=yaml.FullLoader)
        
        if config_type=='object':
            cfg = dotdict(cfg)
        return cfg
    
    experiment_dir = r'C:\Users\Safwen\.spyder-py3\Self Supervised learning\experiments\supervised\sl_without_pretrain_aug'
    config_file=os.path.join(experiment_dir,'config_sl.yaml')
    ckpt_name='resnet18_best.pth'
    ckpt_path=os.path.join(experiment_dir,ckpt_name)
    
    assert os.path.isfile(config_file), "No parameters config file found at {}".format(config_file)

    cfg = load_yaml(config_file,config_type='object')

    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    cfg.use_cuda=use_cuda
    device = torch.device("cuda:{}".format(cfg.cuda_num) if use_cuda else "cpu")


    ## get the dataloaders
    _,_,dloader_test = dataloaders.get_dataloaders(cfg,val_split=.2)
    
    
    # Load the model
    model = models.get_model(cfg)
    state_dict = torch.load(ckpt_path,map_location=device)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    test_loss,test_acc = test(model, device, dloader_test, criterion, experiment_dir)

    print('Test: Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(test_loss, test_acc))
