import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import _create_model_training_folder
import yaml
from torch.utils.data.dataloader import default_collate

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):
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
        
        train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,\
                            shuffle=True,drop_last=False,num_workers=0, collate_fn=default_collate)
        
        #print(train_loader)

        #train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
        #                          num_workers=self.num_workers, drop_last=False, shuffle=True)
        #data, label,idx,_ = next(iter(train_loader))
        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            #for (batch_view_1, batch_view_2),(data, label, _,_) in enumerate(tqdm(train_loader)):
            for (batch_view_1, batch_view_2),label in train_loader:
                
                batch_view_1 = batch_view_1.cuda()
                batch_view_2 = batch_view_2.cuda()
                label=label.to(device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1

            print("End of epoch {}".format(epoch_counter))
            print("loss {}".format(loss))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
