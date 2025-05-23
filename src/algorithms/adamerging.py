# ref: https://github.com/EnnengYang/AdaMerging/tree/main/src
# ref: https://github.com/jzhang538/BadMerging/blob/main/src/main_adamerging_badmergingoff.py

import copy
import random
import torch
import pytorch_lightning as pl 
from torch.nn.utils.stateless import functional_call




def softmax_entropy(x):
    # print(x.size(1))
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class AdaMerging_Task(pl.LightningModule):
    def __init__(self, source, model_list, pre_model_encoder, taskvector_list, lr, train_loader_list):
        super(AdaMerging_Task, self).__init__()
        self.source = source
        self.model_list = model_list
        self.pre_model_encoder = pre_model_encoder
        self.taskvector_list = taskvector_list
        self.ckpt_list = copy.deepcopy(taskvector_list)
        self.raw_lambda = torch.nn.Parameter(torch.full((len(self.model_list),), 0.3), requires_grad = True)
        self.merged_encoder = copy.deepcopy(self.pre_model_encoder)
        self.merged_encoder.eval()
        self.classifiers = torch.nn.ModuleList() 
        self.train_loader_list = train_loader_list
        self.lr = lr
        for param in self.merged_encoder.parameters():
            param.requires_grad = False
        for param in self.pre_model_encoder.parameters():
            param.requires_grad = False
        # add classifier
        
        if(self.source == 'simclr'):
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line41
            for i in range(len(model_list)):
                classifier =  copy.deepcopy(self.model_list[i].model.fc)
                for parm in classifier.parameters():
                    parm.requires_grad = False
                self.classifiers.append(classifier)
        if(self.source == 'hf_vit'):
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line104
            for i in range(len(model_list)):
                classifier =  copy.deepcopy(model_list[i].model.classifier)
                for parm in classifier.parameters():
                    parm.requires_grad = False
                self.classifiers.append(classifier)
        if(self.source == 'hf_clip'):
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line139
            for i in range(len(model_list)):
                task_layers = torch.nn.ModuleDict({
                    'visual_proj': copy.deepcopy(self.model_list[i].model.visual_projection),  
                    'fc': copy.deepcopy(self.model_list[i].classifier)                       
                })
                for parm in task_layers.parameters():
                    parm.requires_grad = False
                self.classifiers.append(task_layers)
        if(self.source == 'hf_bert'):
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line182
            classifier_dropout = (
                copy.deepcopy(self.model_list[0].model.config.classifier_dropout) if self.model_list[0].model.config.classifier_dropout is not None else copy.deepcopy(self.model_list[0].model.config.hidden_dropout_prob)
            )
            self.dropout = torch.nn.Dropout(classifier_dropout)
            self.requires_grad = False
            for i in range(len(model_list)):
                task_layers = torch.nn.ModuleDict({
                    'fc': self.model_list[i].model.classifier 
                })
                for parm in task_layers.parameters():
                    parm.requires_grad = False
                self.classifiers.append(task_layers)

        

    def train_dataloader(self):
        return self.train_loader_list

    def forward(self, x):

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"Trainable: {name}")
        # import pdb; pdb.set_trace()

        weighted_taskvectors = [
            {k: v * lamb for k, v in taskvector.vector.items()}
            for taskvector, lamb in zip(self.taskvector_list, self.raw_lambda)
        ]

        merged_delta = {}
        for name in weighted_taskvectors[0].keys():
            merged_delta[name] = sum([vec[name] for vec in weighted_taskvectors])

        delta = merged_delta
        merged_params = {}
        for name, param in self.pre_model_encoder.named_parameters():
            # print(name)
            merged_params[name] = param + delta[name]
        self.pre_model_encoder.train()
        output =  functional_call(self.pre_model_encoder, (merged_params), (x,))
        if(self.source == 'simclr'):
            # output = self.merged_encoder(x)
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/models/simclr/resnet.py line127
           
            output = output.mean(dim=[2, 3])  
   
            outputs = [classifier(output) for classifier in self.classifiers]
            return outputs
        if(self.source == 'hf_vit'):
            # /home/xxx/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/vit/modeling_vit.py
            sequence_output = output[0]
            outputs = [classifier(sequence_output[:, 0, :]) for classifier in self.classifiers]
            # print(outputs[0].shape)
            return outputs
        if(self.source == 'hf_clip'):
            # from /home/xxx/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/clip/modeling_clip.py # line 1563
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line151
            vision_outputs = output
            pooled_output = vision_outputs[1]  # pooled_output
            outputs = []
            for task_layers in self.classifiers:
                projected_features = task_layers['visual_proj'](pooled_output)
                # print(projected_features.shape)
                # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/ft_class.py line 118
                # from /home/xxx/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/clip/modeling_clip.py # line 1563
                projected_features = projected_features / projected_features.norm(dim=-1, keepdim=True)
                class_output = task_layers['fc'](projected_features)
                outputs.append(class_output)
            return outputs
    
    def training_step(self, train_batch):
        current_loss = 0

        for i, batch in enumerate(train_batch):
 
            x,_ = batch
            outputs = self.forward(x)
            current_loss += softmax_entropy(outputs[i]).mean(0)

        self.log('train_loss', current_loss) 
        self.current_loss= current_loss.detach().cpu().item()
        return current_loss
    
    def on_train_end(self):
        self.last_step_lambda = self.raw_lambda.cpu().tolist()
        self.last_step_loss = self.current_loss
    
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(name)

    # def on_after_backward(self):
    #     print("raw_lambda.grad =", self.raw_lambda.grad)
    # #     print(self.raw_lambda)


    def configure_optimizers(self): 
        optimizer = torch.optim.Adam([self.raw_lambda], lr=self.lr)
        return {'optimizer': optimizer}
    
    def get_merged_ckpt(self):
        self.raw = [float(lamb.item()) for lamb in self.raw_lambda]
        weighted_taskvectors = [
            {k: v * lamb for k, v in taskvector.vector.items()}
            for taskvector, lamb in zip(self.taskvector_list, self.raw)
        ]
        merged_delta = {}
        for name in weighted_taskvectors[0].keys():
            
            merged_delta[name] = sum([vec[name] for vec in weighted_taskvectors])
        merged_params = {
            key: self.merged_encoder.state_dict()[key].to('cuda') + merged_delta[key].to('cuda')
            for key in merged_delta.keys()
        }

        return merged_params
    
   

'''
Type 问题
权重加载时间问题
[70.33, 48.01641586867305, 38.98342663932154] [94.42, 83.99452804377565]
[70.61, 48.48899390623057, 39.13862867911978] [94.69999999999999, 84.16863574182317]
'''        

class AdaMerging_Layer(pl.LightningModule):
    def __init__(self, source, model_list, pre_model_encoder, taskvector_list, lr, train_loader_list):  
        super(AdaMerging_Layer, self).__init__()
        self.source = source
        self.model_list = model_list
        self.pre_model_encoder = pre_model_encoder
        self.taskvector_list = taskvector_list
        self.raw_lambda = torch.nn.Parameter(torch.full((len(self.model_list),len(self.pre_model_encoder.state_dict())), 0.6), requires_grad = True)
    
        self.merged_encoder = copy.deepcopy(self.pre_model_encoder)
        self.classifiers = torch.nn.ModuleList() 
        self.train_loader_list = train_loader_list
        self.lr = lr
        for param in self.merged_encoder.parameters():
            param.requires_grad = False
        for param in self.pre_model_encoder.parameters():
            param.requires_grad = False

        if(self.source == 'simclr'):
             for i in range(len(model_list)):
                classifier =  copy.deepcopy(self.model_list[i].model.fc)
                for parm in classifier.parameters():
                    parm.requires_grad = False
                self.classifiers.append(classifier)
        if(self.source == 'hf_vit'):
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line104
            for i in range(len(model_list)):
                classifier =  copy.deepcopy(model_list[i].model.classifier)
                for parm in classifier.parameters():
                    parm.requires_grad = False
                self.classifiers.append(classifier)

        if(self.source == 'hf_clip'):
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line139
            for i in range(len(model_list)):
                task_layers = torch.nn.ModuleDict({
                    'visual_proj': copy.deepcopy(self.model_list[i].model.visual_projection),  
                    'fc': copy.deepcopy(self.model_list[i].classifier)                        
                })
                for parm in task_layers.parameters():
                    parm.requires_grad = False
                self.classifiers.append(task_layers)


    def train_dataloader(self):
        return self.train_loader_list

    def forward(self, x):

        weighted_taskvectors = [
            {
                k: v * self.raw_lambda[i][j]  
                for j, (k, v) in enumerate(taskvector.vector.items())
            }
            for i, taskvector in enumerate(self.taskvector_list)
        ]
    
        merged_delta = {}
        for name in weighted_taskvectors[0].keys():
            merged_delta[name] = sum([vec[name] for vec in weighted_taskvectors])

        delta = merged_delta
        merged_params = {}
        for name, param in self.pre_model_encoder.named_parameters():
            merged_params[name] = param + delta[name]
        self.pre_model_encoder.train()
        output =  functional_call(self.pre_model_encoder, (merged_params), (x,))
        if(self.source == 'simclr'):
            # output = self.merged_encoder(x)
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/models/simclr/resnet.py line127
 
            output = output.mean(dim=[2, 3])  
            # 多个输出，列表
            outputs = [classifier(output) for classifier in self.classifiers]
            return outputs
        if(self.source == 'hf_vit'):
            # /home/xxx/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/vit/modeling_vit.py
            sequence_output = output[0]
            outputs = [classifier(sequence_output[:, 0, :]) for classifier in self.classifiers]
            # print(outputs[0].shape)
            return outputs
        if(self.source == 'hf_clip'):
            # from /home/xxx/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/clip/modeling_clip.py # line 1563
            # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line151
            vision_outputs = output
            pooled_output = vision_outputs[1]  # pooled_output
            outputs = []
            for task_layers in self.classifiers:
                projected_features = task_layers['visual_proj'](pooled_output)
                # print(projected_features.shape)
                # from /home/xxx/miniconda3/envs/0407/SecureMerge/src/merge/ft_class.py line 118
                # from /home/xxx/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/clip/modeling_clip.py # line 1563
                projected_features = projected_features / projected_features.norm(dim=-1, keepdim=True)
                class_output = task_layers['fc'](projected_features)
                outputs.append(class_output)
            return outputs
    
    def training_step(self, train_batch):
        current_loss = 0
        for i, batch in enumerate(train_batch):
            x,_ = batch
            outputs = self.forward(x)
            current_loss += softmax_entropy(outputs[i]).mean(0)
     
        self.log('train_loss', current_loss)
        self.current_loss= current_loss.detach().cpu().item()
        return current_loss
    def on_train_end(self):
        self.last_step_lambda = self.raw_lambda.cpu().tolist()
        self.last_step_loss = self.current_loss
    
    # def on_after_backward(self):
    #     print("raw_lambda.grad =", self.raw_lambda.grad)
    # #     print(self.raw_lambda)

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:  
    #             print(name)

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam([self.raw_lambda], lr=self.lr)  
        return {'optimizer': optimizer}
    
    def get_merged_ckpt(self):
        # print(self.raw_lambda)
        
        weighted_taskvectors = [
            {
                k: v * float(self.raw_lambda[i][j].item()) 
                for j, (k, v) in enumerate(taskvector.vector.items())
            }
            for i, taskvector in enumerate(self.taskvector_list)
        ]
        merged_delta = {}
        for name in weighted_taskvectors[0].keys():
            merged_delta[name] = sum([vec[name] for vec in weighted_taskvectors])

        delta = merged_delta
        merged_params = {}
        for key, val in self.merged_encoder.state_dict().items():
            merged_params[key] = val.to('cuda') + delta[key].to('cuda')

        return merged_params