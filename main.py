import os
from tqdm import tqdm

import torch
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

print(torch.__version__)
torch.manual_seed(0)

class FlowersDataset(Dataset):
    """ Flowers Dataset Class loader """
    
    def __init__(self,cfg, annotation_file,data_type='train', \
                 transform=None):
        
        """
        Args:
            image_dir (string):  directory with images
            annotation_file (string):  csv/txt file which has the 
                                        dataset labels
            transforms: The trasforms to apply to images
        """
        
        self.data_path = os.path.join(cfg.root_path,cfg.data_path,cfg.imgs_dir)
        self.label_path = os.path.join(cfg.root_path,cfg.data_path,cfg.labels_dir,annotation_file)
        self.transform=transform
        self.pretext = cfg.pretext
        if self.pretext == 'rotation':
            self.num_rot = cfg.num_rot
        self._load_data()

    def _load_data(self):
        '''
        function to load the data in the format of [[img_name_1,label_1],
        [img_name_2,label_2],.....[img_name_n,label_n]]
        '''
        self.labels = pd.read_csv(self.label_path)
        
        self.loaded_data = []
#        self.read_data=[]
        for i in range(self.labels.shape[0]):
            img_name = self.labels['Filename'][i]#os.path.join(self.data_path, self.labels['Category'][i],self.labels['FileName'][i])
            #print(img_name)
            #data.append(io.imread(os.path.join(self.image_dir, self.labels['img_name'][i])))
            label = self.labels['Label'][i]
            img = Image.open(img_name)
            self.loaded_data.append((img,label,img_name))
            img.load()#This closes the image object or else you will get too many open file error
#            self.read_data.append((img,label))

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):

        idx = idx % len(self.loaded_data)
        img,label,img_name = self.loaded_data[idx]
#        img = io.imread(img_name)
#        img = Image.open(img_name)   
        img,label = self._read_data(img,label)
        
        return img,label

    def _read_data(self,img,label):
        
        
            # supervised mode; if in supervised mode define a loader function 
            #that given the index of an image it returns the image and its 
            #categorical label
        img = self.transform(img)
        return img, label

def main():
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
     
    config_path = r'C:\Users\Safwen\.spyder-py3\BYOL\PyTorch-BYOL-master\config\config_sl.yaml'
    cfg = load_yaml(config_path,config_type='object') 
    
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #device='cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])
    transform_2 = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((cfg.img_sz,cfg.img_sz)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
    
                transforms.ToTensor()
                
            ])
    annotation_file = 'data_recognition_train.csv'                                 
    
    train_dataset = FlowersDataset(cfg,annotation_file,\
                            data_type='train',transform=MultiViewDataInjector([transform_2, transform_2]))
    print('train data load success')
    #train_dataset = datasets.STL10('/home/thalles/Downloads/', split='train+unlabeled', download=True,
    #                              transform=MultiViewDataInjector([data_transform, data_transform]))

    # online network
    online_network = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)



if __name__ == '__main__':
    main()