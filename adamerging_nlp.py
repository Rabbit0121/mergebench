# ref: https://github.com/EnnengYang/AdaMerging/tree/main/src
# ref: https://github.com/jzhang538/BadMerging/blob/main/src/main_adamerging_badmergingoff.py

import copy
import torch
import pytorch_lightning as pl 
from torch.nn.utils.stateless import functional_call


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class AdaMerging_Task_NLP(pl.LightningModule):
    def __init__(self, source, model_list, pre_model_encoder, taskvector_list, lr, train_loader_list, config_pad_token_id=None):
        super(AdaMerging_Task_NLP, self).__init__()
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
        if(self.source == 'hf_gpt2'):
            self.config_pad_token_id = config_pad_token_id
        for param in self.merged_encoder.parameters():
            param.requires_grad = False
        for param in self.pre_model_encoder.parameters():
            param.requires_grad = False
        # add classifier
        if(self.source == 'hf_bert'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line182
            classifier_dropout = (
                copy.deepcopy(self.model_list[0].model.config.classifier_dropout) if self.model_list[0].model.config.classifier_dropout is not None else copy.deepcopy(self.model_list[0].model.config.hidden_dropout_prob)
            )
            self.dropout = torch.nn.Dropout(classifier_dropout)
            self.requires_grad = False
            for i in range(len(model_list)):
                task_layers = torch.nn.ModuleDict({
                    'fc': copy.deepcopy(self.model_list[i].model.classifier)  # 分类头
                })
                for parm in task_layers.parameters():
                    parm.requires_grad = False
                self.classifiers.append(task_layers)
        if(self.source == 'hf_roberta'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line225
            for i in range(len(model_list)):
                task_layers = torch.nn.ModuleDict({
                    'fc': copy.deepcopy(self.model_list[i].model.classifier)  # 分类头
                })
                for parm in task_layers.parameters():
                    parm.requires_grad = False
                self.classifiers.append(task_layers)
        if(self.source == 'hf_gpt2'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line261
            for i in range(len(model_list)):
                task_layers = torch.nn.ModuleDict({
                    'fc': copy.deepcopy(self.model_list[i].model.score)  # 分类头
                })
                for parm in task_layers.parameters():
                    parm.requires_grad = False
                self.classifiers.append(task_layers)


        

    def train_dataloader(self):
        return self.train_loader_list

    def forward(self, input_ids, attention_mask):
        weighted_taskvectors = [
            {k: v * lamb for k, v in taskvector.vector.items()}
            for taskvector, lamb in zip(self.taskvector_list, self.raw_lambda)
        ]
        # 合并多个字典（按 key 相加），保持计算图
        merged_delta = {}
        for name in weighted_taskvectors[0].keys():
            merged_delta[name] = sum([vec[name] for vec in weighted_taskvectors])

        delta = merged_delta
        merged_params = {}
        for name, param in self.pre_model_encoder.named_parameters():
            # print(name)
            merged_params[name] = param + delta[name]
        # print(self.pre_model_encoder.state_dict().keys())

        
        if(self.source != 'hf_gpt2'):
            output =  functional_call(self.pre_model_encoder, (merged_params), (input_ids,attention_mask,))
        else:
            output =  functional_call(self.pre_model_encoder, 
                                      (merged_params), args=(input_ids,), 
                                      kwargs={"attention_mask": attention_mask},)
            
        if(self.source == 'hf_bert'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py # 197
            outputs = output
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            outputs = []
            for task_layers in self.classifiers:
                class_output = task_layers['fc'](pooled_output)
                outputs.append(class_output)
            return outputs
        if(self.source == 'hf_roberta'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py # 236
            outputs = output
            sequence_output = outputs[0]
            outputs = []
            for task_layers in self.classifiers:
                class_output = task_layers['fc'](sequence_output)
                outputs.append(class_output)
            return outputs
        if(self.source == 'hf_gpt2'):
            if input_ids is not None:
                batch_size, sequence_length = input_ids.shape[:2]
        
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config_pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(self.pre_model_encoder.device)
        
            transformer_outputs = output
            hidden_states = transformer_outputs[0]
            outputs = []
            for task_layers in self.classifiers:
                logits = task_layers['fc'](hidden_states)
                batch_size, _ = input_ids.shape[:2]
                pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
                outputs.append(pooled_logits)
            return outputs
    
    def training_step(self, train_batch):
        current_loss = 0
        # print(len(train_batch))
        for i, batch in enumerate(train_batch):
            input_ids, attention_mask, _= batch['input_ids'], batch['attention_mask'], batch['label']
            outputs = self.forward(input_ids, attention_mask)
            current_loss += softmax_entropy(outputs[i]).mean(0)
        # print(current_loss)
        self.log('train_loss', current_loss) 
        self.current_loss= current_loss.detach().cpu().item()
        return current_loss
    
    def on_train_end(self):
        self.last_step_lambda = self.raw_lambda.cpu().tolist()
        self.last_step_loss = self.current_loss
    
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:  # 仅打印有梯度的参数（即被更新的参数）
    #             print(name)

    # def on_after_backward(self):
    #     print("raw_lambda.grad =", self.raw_lambda.grad)
    # #     print(self.raw_lambda)


    def configure_optimizers(self): 
        optimizer = torch.optim.Adam([self.raw_lambda], lr=self.lr)  # 仅训练参数
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
    


class AdaMerging_Layer_NLP(pl.LightningModule):
    def __init__(self, source, model_list, pre_model_encoder, taskvector_list, lr, train_loader_list, config_pad_token_id=None):
        super(AdaMerging_Layer_NLP, self).__init__()
        self.source = source
        self.model_list = model_list
        self.pre_model_encoder = pre_model_encoder
        self.taskvector_list = taskvector_list
        self.ckpt_list = copy.deepcopy(taskvector_list)
        self.raw_lambda = torch.nn.Parameter(torch.full((len(self.model_list),len(self.pre_model_encoder.state_dict())), 0.3), requires_grad = True)
        # print(self.raw_lambda.shape)self.merged_encoder = copy.deepcopy(self.pre_model_encoder)
        self.merged_encoder = copy.deepcopy(self.pre_model_encoder)
        self.merged_encoder.eval()
        self.classifiers = torch.nn.ModuleList() 
        self.train_loader_list = train_loader_list
        self.lr = lr
        if(self.source == 'hf_gpt2'):
            self.config_pad_token_id = config_pad_token_id
        for param in self.merged_encoder.parameters():
            param.requires_grad = False
        for param in self.pre_model_encoder.parameters():
            param.requires_grad = False
        # add classifier
        if(self.source == 'hf_bert'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line182
            classifier_dropout = (
                copy.deepcopy(self.model_list[0].model.config.classifier_dropout) if self.model_list[0].model.config.classifier_dropout is not None else copy.deepcopy(self.model_list[0].model.config.hidden_dropout_prob)
            )
            self.dropout = torch.nn.Dropout(classifier_dropout)
            self.requires_grad = False
            for i in range(len(model_list)):
                task_layers = torch.nn.ModuleDict({
                    'fc': self.model_list[i].model.classifier  # 分类头
                })
                for parm in task_layers.parameters():
                    parm.requires_grad = False
                self.classifiers.append(task_layers)
        if(self.source == 'hf_roberta'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line225
            for i in range(len(model_list)):
                task_layers = torch.nn.ModuleDict({
                    'fc': self.model_list[i].model.classifier  # 分类头
                })
                for parm in task_layers.parameters():
                    parm.requires_grad = False
                self.classifiers.append(task_layers)
        if(self.source == 'hf_gpt2'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py #line261
            for i in range(len(model_list)):
                task_layers = torch.nn.ModuleDict({
                    'fc': copy.deepcopy(self.model_list[i].model.score)  # 分类头
                })
                for parm in task_layers.parameters():
                    parm.requires_grad = False
                self.classifiers.append(task_layers)

        

    def train_dataloader(self):
        return self.train_loader_list

    def forward(self, input_ids, attention_mask):
        weighted_taskvectors = [
            {
                k: v * self.raw_lambda[i][j]  # i=模型索引, j=参数索引
                for j, (k, v) in enumerate(taskvector.vector.items())
            }
            for i, taskvector in enumerate(self.taskvector_list)
        ]
        # 合并多个字典（按 key 相加），保持计算图
        merged_delta = {}
        for name in weighted_taskvectors[0].keys():
            merged_delta[name] = sum([vec[name] for vec in weighted_taskvectors])

        delta = merged_delta
        merged_params = {}
        for name, param in self.pre_model_encoder.named_parameters():
            # print(name)
            merged_params[name] = param + delta[name]

        if(self.source != 'hf_gpt2'):
            output =  functional_call(self.pre_model_encoder, (merged_params), (input_ids,attention_mask,))
        else:
            output =  functional_call(self.pre_model_encoder, 
                                      (merged_params), args=(input_ids,), 
                                      kwargs={"attention_mask": attention_mask},)
        if(self.source == 'hf_bert'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py # 197
            outputs = output
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            outputs = []
            for task_layers in self.classifiers:
                class_output = task_layers['fc'](pooled_output)
                outputs.append(class_output)
            return outputs
        if(self.source == 'hf_roberta'):
            # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/merge_class.py # 236
            outputs = output
            sequence_output = outputs[0]
            outputs = []
            for task_layers in self.classifiers:
                class_output = task_layers['fc'](sequence_output)
                outputs.append(class_output)
            return outputs
        if(self.source == 'hf_gpt2'):
            if input_ids is not None:
                batch_size, sequence_length = input_ids.shape[:2]
        
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config_pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(self.pre_model_encoder.device)
        
            transformer_outputs = output
            hidden_states = transformer_outputs[0]
            outputs = []
            for task_layers in self.classifiers:
                logits = task_layers['fc'](hidden_states)
                batch_size, _ = input_ids.shape[:2]
                pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
                outputs.append(pooled_logits)
            return outputs
    
    def training_step(self, train_batch):
        current_loss = 0
        # print(len(train_batch))
        for i, batch in enumerate(train_batch):
            input_ids, attention_mask, _= batch['input_ids'], batch['attention_mask'], batch['label']
            outputs = self.forward(input_ids, attention_mask)
            current_loss += softmax_entropy(outputs[i]).mean(0)
        # print(current_loss)
        self.log('train_loss', current_loss) 
        self.current_loss= current_loss.detach().cpu().item()
        return current_loss
    
    def on_train_end(self):
        self.last_step_lambda = self.raw_lambda.cpu().tolist()
        self.last_step_loss = self.current_loss
    
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:  # 仅打印有梯度的参数（即被更新的参数）
    #             print(name)

    # def on_after_backward(self):
    #     print("raw_lambda.grad =", self.raw_lambda.grad)
    # #     print(self.raw_lambda)


    def configure_optimizers(self): 
        optimizer = torch.optim.Adam([self.raw_lambda], lr=self.lr)  # 仅训练参数
        return {'optimizer': optimizer}
    
    def get_merged_ckpt(self):
        weighted_taskvectors = [
            {
                k: v * float(self.raw_lambda[i][j].item())  # i=模型索引, j=参数索引
                for j, (k, v) in enumerate(taskvector.vector.items())
            }
            for i, taskvector in enumerate(self.taskvector_list)
        ]
        merged_delta = {}
        for name in weighted_taskvectors[0].keys():
            
            merged_delta[name] = sum([vec[name] for vec in weighted_taskvectors])
        merged_params = {
            key: self.merged_encoder.state_dict()[key].to('cuda') + merged_delta[key].to('cuda')
            for key in merged_delta.keys()
        }

        return merged_params
