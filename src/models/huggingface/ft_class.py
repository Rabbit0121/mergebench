import sys

import torchmetrics.classification
sys.path.append('/home/xxx/miniconda3/envs/0407/SecureMerge/src/models/simclr')
import torchmetrics
import pytorch_lightning as pl 
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from transformers import ViTForImageClassification, CLIPVisionModelWithProjection, BertForSequenceClassification, RobertaForSequenceClassification, GPT2ForSequenceClassification
from transformers import get_scheduler
from utils.files import *
pl.seed_everything(42, workers=True)
    
class ViTImageClassifier(pl.LightningModule):
    def __init__(self,  model_path: str, lr: float,max_epochs: int, percent: float, batch_size: int, num_classes:int, result_path: str, devices:int):  
        super().__init__() 
        try:
            self.lr=lr
            self.max_epochs=max_epochs
            self.percent=percent
            self.devices = devices
            self.batch_size=batch_size * self.devices
            self.model_path = model_path
            self.result_path = result_path
            self.save_hyperparameters()
            self.model = ViTForImageClassification.from_pretrained(model_path + '/model', num_labels = num_classes)
            # with torch.no_grad():
            #     torch.nn.init.constant_(self.model.classifier.weight, 1.0)
            # 
            #     torch.nn.init.constant_(self.model.classifier.bias, 0.0)
            # stamfordcars
            with torch.no_grad():
                pl.seed_everything(42, workers=True)
                torch.nn.init.kaiming_normal_(
                    self.model.classifier.weight
                )
                torch.nn.init.zeros_(self.model.classifier.bias)
            self.model_path = model_path
            self.test_accuracy_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
            self.test_accuracy_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
            # print(self.model)
            
        except:
            raise ValueError(f'Invalid pth_path: ') 


    def forward(self, x):    
        logits = self.model(x).logits
        # print(logits)
        return logits

    def training_step(self, train_batch):  
        x, y = train_batch  
        logits = self.forward(x)  
        loss = F.cross_entropy(logits, y)
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
        data = {"model":self.model_path,"lr":self.lr, "max_epochs":self.max_epochs, "batch_size":self.batch_size, "percent":self.percent,
                "top1_acc":self.test_accuracy_top1.compute().item(), "top5_acc":self.test_accuracy_top5.compute().item()}
        write_to_csv(self.result_path,headers, data)

    def configure_optimizers(self): 
        # optimizer = torch.optim.Adam([param for _, param in self.named_parameters() if param.requires_grad], lr=self.lr)  # 使用所有可训练参数
        # return {'optimizer': optimizer}

        #stanfordcars use this
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}



  

class CLIPImageClassifier(pl.LightningModule):
    def __init__(self,  model_path: str, lr: float,max_epochs: int, percent: float, batch_size: int, num_classes:int, result_path: str, devices:int):  
        super().__init__() 
        try:
            self.lr=lr
            self.max_epochs=max_epochs
            self.percent=percent
            self.devices = devices
            self.batch_size=batch_size * self.devices
            self.model_path = model_path
            self.result_path = result_path
            self.save_hyperparameters()
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_path + '/model')
            self.model.visual_projection.weight.requires_grad = False
            self.classifier = torch.nn.Linear(self.model.visual_projection.out_features, num_classes, bias=False)
            self.classifier.weight.requires_grad = False
            self.model_path = model_path
            self.test_accuracy_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
            self.test_accuracy_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
            # print(self.model)
            
        except:
            raise ValueError(f'Invalid pth_path: ') 


    def forward(self, x):    
        output = self.model(x)
        # print(output[0].shape)
        output = output[0] / output[0].norm(dim=-1, keepdim=True)
        logits = self.classifier(output)
        # print(logits.shape)
        return logits

    def training_step(self, train_batch):  
        x, y = train_batch  
        logits = self.forward(x)  
        loss = F.cross_entropy(logits, y)
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
        data = {"model":self.model_path,"lr":self.lr, "max_epochs":self.max_epochs, "batch_size":self.batch_size, "percent":self.percent,
                "top1_acc":self.test_accuracy_top1.compute().item(), "top5_acc":self.test_accuracy_top5.compute().item()}
        write_to_csv(self.result_path,headers, data)

    def configure_optimizers(self): 

        optimizer = torch.optim.Adam([param for name, param in self.named_parameters() if param.requires_grad and  'model.visual_projection.weight' not in name], lr=self.lr)
        return {'optimizer': optimizer}



class BertSequenceClassifier(pl.LightningModule):
    def __init__(self,  model_path: str, lr: float,max_epochs: int, percent: float, batch_size: int, num_classes:int, result_path: str, devices:int):  
        super().__init__() 
        try:
            self.lr=lr
            self.max_epochs=max_epochs
            self.percent=percent
            self.devices = devices
            self.batch_size=batch_size * self.devices
            self.model_path = model_path
            self.result_path = result_path
            self.num_classes = num_classes
            self.save_hyperparameters()
            self.model = BertForSequenceClassification.from_pretrained(model_path + '/model', num_labels = num_classes)
            # print(self.model)
            # import pdb; pdb.set_trace()
            with torch.no_grad():
           
                torch.nn.init.constant_(self.model.classifier.weight, 1.0)
   
                torch.nn.init.constant_(self.model.classifier.bias, 0.0)
            # import pdb; pdb.set_trace()
            self.model_path = model_path
            # acc
            self.test_accuracy_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
            # f1
            if(num_classes == 2):
                self.test_f1 = torchmetrics.classification.BinaryF1Score()
                # matthewscoef
                self.test_mcc = torchmetrics.classification.BinaryMatthewsCorrCoef()
            
        except:
            raise ValueError(f'Invalid pth_path: ') 


    def forward(self, input_ids, attention_mask=None):

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits


    def training_step(self, train_batch):
        input_ids, attention_mask, labels = train_batch['input_ids'], train_batch['attention_mask'], train_batch['label']  
        logits = self.forward(input_ids, attention_mask) 
        loss = F.cross_entropy(logits, labels)  
        self.log('train_loss', loss) 
    
        return loss  
    
    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)

    def test_step(self, test_batch):
        input_ids, attention_mask, labels = test_batch['input_ids'], test_batch['attention_mask'], test_batch['label']  
        logits = self.forward(input_ids, attention_mask) 
        self.test_accuracy_top1.update(logits, labels)
        self.log('top1_acc', self.test_accuracy_top1)
        if(self.num_classes == 2):
            preds = torch.argmax(logits, dim=1)  
            self.test_f1.update(preds, labels)
            self.test_mcc.update(preds, labels)
            self.log('f1', self.test_f1)
            self.log('mcc', self.test_mcc)
    
    def on_test_epoch_end(self):
        if(self.num_classes == 2):
            headers = ["model","lr", "max_epochs", "batch_size", "percent", "top1_acc", "f1", "mcc" ]
            data = {"model":self.model_path,"lr":self.lr, "max_epochs":self.max_epochs, "batch_size":self.batch_size, "percent":self.percent,
                    "top1_acc":self.test_accuracy_top1.compute().item(), "f1": self.test_f1.compute().item(), "mcc": self.test_mcc.compute().item()}
            write_to_csv(self.result_path,headers, data)
        else:
            headers = ["model","lr", "max_epochs", "batch_size", "percent", "top1_acc"]
            data = {"model":self.model_path,"lr":self.lr, "max_epochs":self.max_epochs, "batch_size":self.batch_size, "percent":self.percent,
                    "top1_acc":self.test_accuracy_top1.compute().item()}
            write_to_csv(self.result_path,headers, data)
            

    def configure_optimizers(self):  

        optimizer = torch.optim.AdamW([param for _, param in self.named_parameters() if param.requires_grad], lr=self.lr)  
        num_training_steps = self.trainer.estimated_stepping_batches  

        num_warmup_steps = int(0.1 * num_training_steps)

        lr_scheduler = get_scheduler(
            name="linear", 
            optimizer=optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  
                "frequency": 1
            }
        }

class RobertaSequenceClassifier(pl.LightningModule):
    def __init__(self,  model_path: str, lr: float,max_epochs: int, percent: float, batch_size: int, num_classes:int, result_path: str, devices:int):  
        super().__init__() 
        try:
            self.lr=lr
            self.max_epochs=max_epochs
            self.percent=percent
            self.devices = devices
            self.batch_size=batch_size * self.devices
            self.model_path = model_path
            self.result_path = result_path
            self.num_classes = num_classes
            self.save_hyperparameters()
            self.model = RobertaForSequenceClassification.from_pretrained(model_path + '/model', num_labels = num_classes)
            # print(self.model)
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                pl.seed_everything(42, workers=True)
                torch.nn.init.kaiming_normal_(
                    self.model.classifier.dense.weight
                )
                torch.nn.init.zeros_(self.model.classifier.dense.bias)
                torch.nn.init.kaiming_normal_(
                    self.model.classifier.out_proj.weight
                )
                torch.nn.init.zeros_(self.model.classifier.out_proj.bias)
            # import pdb; pdb.set_trace()
            self.model_path = model_path
            # acc
            self.test_accuracy_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
            # f1
            if(num_classes == 2):
                self.test_f1 = torchmetrics.classification.BinaryF1Score()
                # matthewscoef
                self.test_mcc = torchmetrics.classification.BinaryMatthewsCorrCoef()
            
        except:
            raise ValueError(f'Invalid pth_path: ') 


    def forward(self, input_ids, attention_mask=None):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits


    def training_step(self, train_batch):
        input_ids, attention_mask, labels = train_batch['input_ids'], train_batch['attention_mask'], train_batch['label']  u
        logits = self.forward(input_ids, attention_mask)  
        loss = F.cross_entropy(logits, labels)  
        self.log('train_loss', loss)
        # print("logits",logits.shape) 
        return loss  
    
    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)

    def test_step(self, test_batch):
        input_ids, attention_mask, labels = test_batch['input_ids'], test_batch['attention_mask'], test_batch['label'] 
        logits = self.forward(input_ids, attention_mask) 
        self.test_accuracy_top1.update(logits, labels)
        self.log('top1_acc', self.test_accuracy_top1)
        if(self.num_classes == 2):
            preds = torch.argmax(logits, dim=1)  
            self.test_f1.update(preds, labels)
            self.test_mcc.update(preds, labels)
            self.log('f1', self.test_f1)
            self.log('mcc', self.test_mcc)
    
    def on_test_epoch_end(self):
        if(self.num_classes == 2):
            headers = ["model","lr", "max_epochs", "batch_size", "percent", "top1_acc", "f1", "mcc" ]
            data = {"model":self.model_path,"lr":self.lr, "max_epochs":self.max_epochs, "batch_size":self.batch_size, "percent":self.percent,
                    "top1_acc":self.test_accuracy_top1.compute().item(), "f1": self.test_f1.compute().item(), "mcc": self.test_mcc.compute().item()}
            write_to_csv(self.result_path,headers, data)
        else:
            headers = ["model","lr", "max_epochs", "batch_size", "percent", "top1_acc"]
            data = {"model":self.model_path,"lr":self.lr, "max_epochs":self.max_epochs, "batch_size":self.batch_size, "percent":self.percent,
                    "top1_acc":self.test_accuracy_top1.compute().item()}
            write_to_csv(self.result_path,headers, data)

    def configure_optimizers(self):  
        optimizer = torch.optim.AdamW([param for _, param in self.named_parameters() if param.requires_grad], lr=self.lr)  
        num_training_steps = self.trainer.estimated_stepping_batches  

        num_warmup_steps = int(0.1 * num_training_steps)
  
        lr_scheduler = get_scheduler(
            name="linear", 
            optimizer=optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  
                "frequency": 1
            }
        }
    
class GPT2SequenceClassifier(pl.LightningModule):
    def __init__(self,  model_path: str, lr: float,max_epochs: int, percent: float, batch_size: int, num_classes:int, result_path: str, devices:int):  
        super().__init__() 
        try:
            self.lr=lr
            self.max_epochs=max_epochs
            self.percent=percent
            self.devices = devices
            self.batch_size=batch_size * self.devices
            self.model_path = model_path
            self.result_path = result_path
            self.num_classe = num_classes
            self.save_hyperparameters()
            self.model = GPT2ForSequenceClassification.from_pretrained(model_path + '/model', num_labels = self.num_classe)
            # print(self.model)
            # import pdb; pdb.set_trace()
            with torch.no_grad():
    
                torch.nn.init.constant_(self.model.score.weight, 1.0)
   
                # torch.nn.init.constant_(self.model.score.bias, 0.0)
            # import pdb; pdb.set_trace()
            self.model_path = model_path
            # acc
            self.test_accuracy_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
            # f1
            if (self.num_classe == 2):
                self.test_f1 = torchmetrics.classification.BinaryF1Score()
                # matthewscoef
                self.test_mcc = torchmetrics.classification.BinaryMatthewsCorrCoef()
            
        except:
            raise ValueError(f'Invalid pth_path: ') 


    def forward(self, input_ids, attention_mask=None):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits


    def training_step(self, train_batch):
        input_ids, attention_mask, labels = train_batch['input_ids'], train_batch['attention_mask'], train_batch['label']  
        logits = self.forward(input_ids, attention_mask) 
        loss = F.cross_entropy(logits, labels)  
        self.log('train_loss', loss)  
        return loss  
    
    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)

    def test_step(self, test_batch):
        input_ids, attention_mask, labels = test_batch['input_ids'], test_batch['attention_mask'], test_batch['label']  
        logits = self.forward(input_ids, attention_mask) 
        self.test_accuracy_top1.update(logits, labels)
        self.log('top1_acc', self.test_accuracy_top1)
        if (self.num_classe == 2):
            preds = torch.argmax(logits, dim=1)
            self.test_f1.update(preds, labels)
            self.test_mcc.update(preds, labels)
            self.log('f1', self.test_f1)
            self.log('mcc', self.test_mcc)
    
    def on_test_epoch_end(self):
        if(self.num_classe == 2):
            headers = ["model","lr", "max_epochs", "batch_size", "percent", "top1_acc", "f1", "mcc" ]
            data = {"model":self.model_path,"lr":self.lr, "max_epochs":self.max_epochs, "batch_size":self.batch_size, "percent":self.percent,
                    "top1_acc":self.test_accuracy_top1.compute().item(), "f1": self.test_f1.compute().item(), "mcc": self.test_mcc.compute().item()}
            write_to_csv(self.result_path,headers, data)
        else:
            headers = ["model","lr", "max_epochs", "batch_size", "percent", "top1_acc"]
            data = {"model":self.model_path,"lr":self.lr, "max_epochs":self.max_epochs, "batch_size":self.batch_size, "percent":self.percent,
                    "top1_acc":self.test_accuracy_top1.compute().item()}
            write_to_csv(self.result_path,headers, data)

    def configure_optimizers(self):  
        optimizer = torch.optim.AdamW([param for _, param in self.named_parameters() if param.requires_grad], lr=self.lr)  
        num_training_steps = self.trainer.estimated_stepping_batches  

        num_warmup_steps = int(0.1 * num_training_steps)
  
        lr_scheduler = get_scheduler(
            name="linear", 
            optimizer=optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  
                "frequency": 1
            }
        }