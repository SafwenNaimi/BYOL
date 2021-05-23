import os,sys
import numpy as np
from tqdm import tqdm
from itertools import islice
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader 
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import shutil
from simclr_tran import TransformsSimCLR

from PIL import Image
import matplotlib.pyplot as plt

#from lshash.lshash import LSHash
import logging

import utils
#from models import get_model
from d_loaders import get_dataloaders
from evaluate import validate,test

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

torch.set_default_tensor_type(dtype)
#%%

def save_yaml(config,save_path='config.yaml'):
    if type(config)!=dict:
        config=dict(config)
    with open(save_path, 'w') as file:
        yaml.dump(config, file)
        
def save_checkpoint(state, is_best, save_path, checkpoint='checkpoint.pth', best_model='model_best.pth'):
    """ Save model. """
    os.makedirs(save_path, exist_ok=True)
    torch.save(state, save_path + '/' + checkpoint)
    if is_best:
        shutil.copyfile(save_path + '/' + checkpoint, save_path + '/' + best_model)

def load_checkpoint(checkpoint, model, device='cuda',optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint,map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'],strict=False)
    except:
        model.load_state_dict(checkpoint,strict=False)

#    state = torch.load(path, map_location='cuda:0')
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
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

def get_model(params,pretrained=False):
    if params.network=='resnet18':
        model = models.resnet18(pretrained=pretrained)
        if params.pretext=='rotation':
            params.num_classes=params.num_rot
        model.fc = nn.Linear(in_features=model.fc.in_features,out_features=params.num_classes,bias=True)
        return model

def load_checkpoint(model,checkpoint_path,device):
    pass

class LogisticRegression(nn.Module):
    
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)

def train(epoch, model, device, dataloader, optimizer, scheduler, criterion, experiment_dir, writer):
    """ Train loop, predict rotations. """
    global iter_cnt
#    progbar = tqdm(total=len(dataloader), desc='Train')
    progbar = tqdm(total=10, desc='Train')

    loss_record = RunningAverage()
    acc_record = RunningAverage()
    correct=0
    total=0
    save_path = experiment_dir + '/'
    os.makedirs(save_path, exist_ok=True)
    model.train()
 #   for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
    for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
        data, label = data.to(device), label.to(device)
        #optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        
        # measure accuracy and record loss
        confidence, predicted = output.max(1)
        correct += predicted.eq(label).sum().item()
        #acc = utils.compute_acc(output, label)
        total+=label.size(0)
        acc = correct/total
        
        acc_record.update(100*acc)
        loss_record.update(loss.item())

        writer.add_scalar('train/Loss_batch', loss.item(), iter_cnt)
        writer.add_scalar('train/Acc_batch', acc, iter_cnt)
        iter_cnt+=1

#        logging.info('Train Step: {}/{} Loss: {:.4f}; Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progbar.set_description('Train (loss=%.4f)' % (loss_record()))
        progbar.update(1)
        
    if scheduler:  
        scheduler.step()
        
    LR=optimizer.param_groups[0]['lr']


    writer.add_scalar('train/Loss_epoch', loss_record(), epoch)
    writer.add_scalar('train/Acc_epoch', acc_record(), epoch)
    logging.info('Train Epoch: {} LR: {:.4f} Avg Loss: {:.4f}; Avg Acc: {:.4f}'.format(epoch,LR, loss_record(), acc_record()))

    return loss_record,acc_record

#%%  
def train_and_evaluate(cfg):
    
    
#    correct = (output == targets).float().sum()
#    num_samples=output.shape[0]
#    return correct/num_samples
    def set_logger(log_path):
    
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)
        
        # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)
        
    
    #Training settings
    experiment_dir = os.path.join('experiments',cfg.exp_type,cfg.save_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
    set_logger(os.path.join(experiment_dir,cfg.log))
    logging.info('-----------Starting Experiment------------')
    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    cfg.use_cuda=use_cuda
    device = torch.device("cuda:{}".format(cfg.cuda_num) if use_cuda else "cpu")
    # initialize the tensorbiard summary writer
    
    logs=os.path.join('experiments',cfg.exp_type,'tboard_4')
    writer = SummaryWriter(logs + '/Resnet_FineTune_LAyer1')

    ## get the dataloaders
    dloader_train,dloader_val,dloader_test = get_dataloaders(cfg)
    
    # Load the model
    model = get_model(cfg)
    
    if cfg.use_pretrained:
        pretrained_path =  os.path.join('experiments','supervised',cfg.pretrained_dir,cfg.pretrained_weights)
        state_dict = torch.load(pretrained_path,map_location=device)
        #state_dict=state_dict['state_dict']
        #print(state_dict.keys())
        model.load_state_dict(state_dict, strict=False)
        logging.info('loading pretrained_weights {}'.format(cfg.pretrained_weights))

    if cfg.use_ssl:
        ssl_exp_dir = os.path.join('experiments',\
                                        'self-supervised',cfg.ssl_pretrained_exp_path)
        
        #state_dict = torch.load(os.path.join(ssl_exp_dir,cfg.ssl_weight),\
        #                        map_location=device)
        state_dict = torch.load('experiments/self-supervised/model-final.pth',\
                                map_location=device)
        
        #logging.info('loading pretrained_model {}'.format(cfg.ssl_pretrained_exp_path))
        # the stored dict has 3 informations - epoch,state_dict and optimizer
        #print(state_dict.keys())
        #print(state_dict['online_network_state_dict'].keys())
        #state_dict=state_dict['online_network_state_dict']
        
        #del state_dict['projetion.net.3.weight']
        #del state_dict['projetion.net.3.bias']
        
        del state_dict['fc.weight']
        del state_dict['fc.bias']

        #del state_dict['encoder.7.0.conv1.weight']
        #del state_dict['encoder.7.0.conv2.weight']
        #del state_dict['encoder.7.1.conv1.weight']
        #del state_dict['encoder.7.1.conv2.weight']
        
        del state_dict['layer4.0.conv1.weight']
        del state_dict['layer4.0.conv2.weight']
        del state_dict['layer4.1.conv1.weight']
        del state_dict['layer4.1.conv2.weight']
        
        #del state_dict['encoder.6.0.conv1.weight']
        #del state_dict['encoder.6.0.conv2.weight']
        #del state_dict['encoder.6.1.conv1.weight']
        #del state_dict['encoder.6.1.conv2.weight']

        del state_dict['layer3.0.conv1.weight']
        del state_dict['layer3.0.conv2.weight']
        del state_dict['layer3.1.conv1.weight']
        del state_dict['layer3.1.conv2.weight']
        
        #del state_dict['encoder.5.0.conv1.weight']
        #del state_dict['encoder.5.0.conv2.weight']
        #del state_dict['encoder.5.1.conv1.weight']
        #del state_dict['encoder.5.1.conv2.weight']


        del state_dict['layer2.0.conv1.weight']
        del state_dict['layer2.0.conv2.weight']
        del state_dict['layer2.1.conv1.weight']
        del state_dict['layer2.1.conv2.weight']
        
        del state_dict['layer1.0.conv1.weight']
        del state_dict['layer1.0.conv2.weight']
        del state_dict['layer1.1.conv1.weight']
        del state_dict['layer1.1.conv2.weight']
    
        model.load_state_dict(state_dict, strict=False)
    
        # Only finetune fc layer
        #layers_list=['fc','avgpool','layer3.0.conv']#,'layer3.1.conv','layer4.0.conv','layer4.1.conv']
        #params_update=[]
        for name, param in model.named_parameters():
            #print(name)
            #for l in layers_list:
                if 'fc'  or'layer4.0.conv' or 'layer4.1.conv'  or 'layer3.0.conv' or 'layer3.1.conv' or 'layer2.0.conv' or 'layer2.1.conv' or 'layer1.0.conv' or 'layer1.1.conv' in name: #or 'layer3.0.conv' or 'layer3.1.conv' or'layer4.0.conv' or 'layer4.1.conv' in name:
                    param.requires_grad = True
###                    print(name)
                else:
                    param.requires_grad = False
#                    print(name)
                   # params_update.append(param)
#            print(param.requires_grad)
    
    model = model.to(device)

    images,_  = next(iter(dloader_train))
    images = images.to(device)
    writer.add_graph(model, images)

    # follow the same setting as RotNet paper
    #model.parameters()
    if cfg.opt=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=float(cfg.lr), momentum=float(cfg.momentum), weight_decay=0.05, nesterov=True)#0.05
    elif cfg.opt=='adam':
        optimizer = optim.Adam(model.parameters(), lr=float(cfg.lr))#, momentum=float(cfg.momentum), weight_decay=5e-4, nesterov=True)

    if cfg.scheduler:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    else:
        scheduler=None
    criterion = nn.CrossEntropyLoss()
    
    global iter_cnt
    iter_cnt=0
    best_loss = 1000
    for epoch in range(cfg.num_epochs):
        
#        print('\nTrain for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        logging.info('\nTrain for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        train_loss,train_acc = train(epoch, model, device, dloader_train, optimizer, scheduler, criterion, experiment_dir, writer)
        
        # validate after every epoch
#        print('\nValidate for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        logging.info('\nValidate for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        val_loss,val_acc = validate(epoch, model, device, dloader_val, criterion, experiment_dir, writer)
        logging.info('Val Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, val_loss, val_acc))
        
       # for name, weight in model.named_parameters():
        #    writer.add_histogram(name,weight, epoch)
         #   writer.add_histogram(f'{name}.grad',weight.grad, epoch)
            
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if epoch % cfg.save_intermediate_weights==0 or is_best:
            save_checkpoint({'Epoch': epoch,'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()}, 
                                    is_best, experiment_dir, checkpoint='{}_epoch{}_checkpoint.pth'.format( cfg.network.lower(),str(epoch)),\
                                    
                                    best_model='{}_best.pth'.format(cfg.network.lower())
                                    )
    writer.close()
    
#    print('\nEvaluate on test')
    logging.info('\nEvaluate test result on best ckpt')
    state_dict = torch.load(os.path.join(experiment_dir,'{}_best.pth'.format(cfg.network.lower())),\
                                map_location=device)
    model.load_state_dict(state_dict, strict=False)

    test_loss,test_acc = test(model, device, dloader_test, criterion, experiment_dir)
    logging.info('Test: Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(test_loss, test_acc))

    # save the configuration file within that experiment directory
    save_yaml(cfg,save_path=os.path.join(experiment_dir,'config_sl.yaml'))
    logging.info('-----------End of Experiment------------')
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
    
    config_file='config/config_sl.yaml'
    cfg = load_yaml(config_file,config_type='object')
    train_and_evaluate(cfg)
    
