import sys
sys.path.append('/home/xxx/miniconda3/envs/0407/SecureMerge/src/models/simclr')
import torchmetrics
from resnet import get_resnet, name_to_params
import pytorch_lightning as pl 
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

from utils.files import *
pl.seed_everything(42, workers=True)

class ImageClassifier(pl.LightningModule):
    def __init__(self, lr: float, max_epochs: int, percent: float, batch_size: int, model_path:str, result_path:str, num_classes:int, devices:int):  
        super().__init__()
        try:
            self.lr=lr
            self.max_epochs=max_epochs
            self.percent=percent
            self.pth_path = model_path
            self.result_path = result_path
            self.num_classes = num_classes
            self.devices = devices
            self.batch_size=batch_size * self.devices
            model, _ = get_resnet(*name_to_params(self.pth_path))
            self.model = model
            self.model.load_state_dict(torch.load(self.pth_path)['resnet'])
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
            if isinstance(self.model.fc, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(self.model.fc.weight)
            if self.model.fc.bias is not None:
                torch.nn.init.constant_(self.model.fc.bias, 0)
            # save pretrain model
            # torch.save(self.model, self.logger'')
            self.test_accuracy_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1)
            self.test_accuracy_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=5)
            self.save_hyperparameters()
        except:
            raise ValueError(f'Invalid pth_path: {self.pth_path}') 

    def forward(self, x):    
        return self.model(x, apply_fc=True)  
  
    def training_step(self, train_batch):  
        x, y = train_batch  
        logits = self.forward(x)  
        loss = F.cross_entropy(logits, y)
        # print('train_loss', loss)
        self.log('train_loss', loss)  
        return loss  
    
    def validation_step(self, val_batch):  
        x, y = val_batch
        logits = self.forward(x) 
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)

    def test_step(self, test_batch):
        x, y = test_batch
        logits = self.forward(x)
        self.test_accuracy_top1.update(logits, y)
        self.test_accuracy_top5.update(logits, y)
        self.log('top1_acc', self.test_accuracy_top1)
        self.log('top5_acc', self.test_accuracy_top5)
    
    def on_test_epoch_end(self):
        headers = ["model","lr", "max_epochs", "batch_size", "percent", "top1_acc", "top5_acc"]
        data = {"model":self.pth_path,"lr":self.lr, "max_epochs":self.max_epochs, "batch_size":self.batch_size, "percent":self.percent,
                "top1_acc":self.test_accuracy_top1.compute().item(), "top5_acc":self.test_accuracy_top5.compute().item()}
        write_to_csv(self.result_path, headers, data)

    def configure_optimizers(self):  
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
       
    