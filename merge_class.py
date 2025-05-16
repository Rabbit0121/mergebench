import copy
from typing import List, Type
import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys
sys.path.append('..')
from models.huggingface.clip.ft_class import CLIPImageClassifier
from models.simclr.ft_class import ImageClassifier 
from models.huggingface.vit.ft_class import ViTImageClassifier
# class merged_model():
#     def __init__(self, model_list: List, classification_task: bool, single_task:bool):
#         self.classification_task = classification_task
#         self.single_task = single_task
#         if(classification_task and single_task):
#             self.classifiers = nn.ModuleList()
#             self.model = get_encoder(model_list[0])
#             for i in range(len(model_list)):
#                 # Assuming each model's `fc` layer (or `classifier`) can be replaced
#                 classifier = get_fc(model_list[i])
#                 self.classifiers.append(classifier)
#         else:
#             self.model = copy.deepcopy(model_list[0])

#     def forward(self, x):
#         output = self.model(x)
#         if(self.classification_task and self.single_task):
#             outputs = [classifier(output) for classifier in self.classifiers]
#             return outputs
#         else: return output


class simclr_merged_model(nn.Module):
    def __init__(self, model_list, single_task:bool):
        super(simclr_merged_model, self).__init__()
        self.single_task = single_task
        self.classifiers = nn.ModuleList()
        self.model_list = model_list
        if(single_task == False):
            self.model_encoder = copy.deepcopy(self.model_list[0].model.net)
            for i in range(len(model_list)):
                classifier =  copy.deepcopy(self.model_list[i].model.fc)
                # print(classifier.state_dict().values())
                self.classifiers.append(classifier)
        else:
            self.model = copy.deepcopy(self.model_list[0])

    def forward(self, x):
        
        if(self.single_task == False):
            output = self.model_encoder(x)
            # 代码来源于 /home/whj/miniconda3/envs/0407/SecureMerge/src/models/simclr/resnet.py line127
            # 对特征进行平均池化，然后改维度
            output = output.mean(dim=[2, 3])  
            # 多个输出，列表
            outputs = [classifier(output) for classifier in self.classifiers]
            return outputs
        else: return self.model(x)
    
    def load_weight(self, merged_checkpoint:dict):
        if(self.single_task == False):
            self.model_encoder.load_state_dict(merged_checkpoint)
        else: self.model.load_state_dict(merged_checkpoint)

# class vit_merged_model(nn.Module):
#     def __init__(self, model_list: List[Type[ViTImageClassifier]], single_task:bool):
#         super(vit_merged_model, self).__init__()
#         self.single_task = single_task
#         self.classifiers = nn.ModuleList() 
#         self.model_list_1 = copy.deepcopy(model_list)
#         self.model_list_2 = copy.deepcopy(model_list)
#         if(single_task == False):
#             self.model_encoder = self.model_list_1[0].model
#             for i in range(len(model_list)):
#                 classifier =  self.model_list_2[i].fc
#                 # print(classifier.state_dict().values())
#                 self.classifiers.append(classifier)
#         else:
#             self.model = copy.deepcopy(model_list[0])

#     def forward(self, x):
        
#         if(self.single_task == False):
#             # from /home/whj/miniconda3/envs/0407/SecureMerge/src/models/huggingface/vit/ft_class.py#line44
#             outputs = self.model_encoder(x)
#             # print(outputs): last_hidden_state, pooler_output
#             # cls
#             outputs = outputs[0][:, 0, :]
#             outputs = [classifier(outputs) for classifier in self.classifiers]
#             return outputs
#         else: return self.model(x)
    
#     def load_weight(self, merged_checkpoint:dict):
#         if(self.single_task == False):
#             self.model_encoder.load_state_dict(merged_checkpoint)
#         else: self.model.load_state_dict(merged_checkpoint)

class vit_merged_model(nn.Module):
    def __init__(self, model_list:List, single_task:bool):
        super(vit_merged_model, self).__init__()
        self.single_task = single_task
        self.classifiers = nn.ModuleList() 
        if(single_task == False):
            self.model_encoder = copy.deepcopy(model_list[0].model.vit)
            for i in range(len(model_list)):
                classifier =  copy.deepcopy(model_list[i].model.classifier)
                # print(classifier.state_dict().values())
                self.classifiers.append(classifier)
        else:
            self.model = copy.deepcopy(model_list[0])

    def forward(self, x):
        
        if(self.single_task == False):
            outputs = self.model_encoder(x)
            # /home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/vit/modeling_vit.py
            sequence_output = outputs[0]
            outputs = [classifier(sequence_output[:, 0, :]) for classifier in self.classifiers]
            # print(outputs[0].shape)
            return outputs
        else: return self.model(x)
    
    def load_weight(self, merged_checkpoint:dict):
        if(self.single_task == False):
            self.model_encoder.load_state_dict(merged_checkpoint)
        else: self.model.load_state_dict(merged_checkpoint)



class clip_merged_model(nn.Module):
    def __init__(self, model_list: List[Type[CLIPImageClassifier]], single_task:bool):
        super(clip_merged_model, self).__init__()
        self.single_task = single_task
        self.classifiers = nn.ModuleList() 
        self.model_list_1 = copy.deepcopy(model_list)
        self.model_list_2 = copy.deepcopy(model_list)
        if(single_task == False):
            self.model_encoder = self.model_list_1[0].model.vision_model
            for i in range(len(model_list)):
                task_layers = nn.ModuleDict({
                    'visual_proj': self.model_list_2[i].model.visual_projection,  # 视觉投影层
                    'fc': self.model_list_2[i].classifier                           # 分类头
                })
                self.classifiers.append(task_layers)
        else:
            self.model = copy.deepcopy(model_list[0])

    def forward(self, x):
        
        if(self.single_task == False):
            # from /home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/clip/modeling_clip.py # line 1563
            vision_outputs = self.model_encoder(x)
            pooled_output = vision_outputs[1]  # pooled_output
            outputs = []
            for task_layers in self.classifiers:
                projected_features = task_layers['visual_proj'](pooled_output)
                # print(projected_features.shape)
                # from /home/whj/miniconda3/envs/0407/SecureMerge/src/merge/ft_class.py line 118
                # from /home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/clip/modeling_clip.py # line 1563
                projected_features = projected_features / projected_features.norm(dim=-1, keepdim=True)
                class_output = task_layers['fc'](projected_features)
                outputs.append(class_output)
            return outputs
        else: return self.model(x)
    
    def load_weight(self, merged_checkpoint:dict):
        if(self.single_task == False):
            self.model_encoder.load_state_dict(merged_checkpoint)
        else: self.model.load_state_dict(merged_checkpoint)

class bert_merged_model(nn.Module):
    def __init__(self, model_list: List, single_task:bool):
        super(bert_merged_model, self).__init__()
        self.single_task = single_task
        self.classifiers = nn.ModuleList() 
        self.model_list_1 = copy.deepcopy(model_list)
        self.model_list_2 = copy.deepcopy(model_list)
        if(single_task == False):
            self.model_encoder = self.model_list_1[0].model.bert
            # from /home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/bert/modeling_bert.py #line 1630
            classifier_dropout = (
                self.model_list_1[0].model.config.classifier_dropout if self.model_list_1[0].model.config.classifier_dropout is not None else self.model_list_1[0].model.config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            for i in range(len(model_list)):
                task_layers = nn.ModuleDict({
                    'fc': self.model_list_2[i].model.classifier  # 分类头
                })
                self.classifiers.append(task_layers)
        else:
            self.model = copy.deepcopy(model_list[0])

    def forward(self, input_ids, attention_mask):
        if(self.single_task == False):
            # from /home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/bert/modeling_bert.py #line 1680
            outputs = self.model_encoder(input_ids, attention_mask)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            # from /home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/clip/modeling_clip.py # line 1563
            outputs = []
            for task_layers in self.classifiers:
                class_output = task_layers['fc'](pooled_output)
                outputs.append(class_output)
            return outputs
        else: return self.model(input_ids, attention_mask)
    
    def load_weight(self, merged_checkpoint:dict):
        if(self.single_task == False):
            self.model_encoder.load_state_dict(merged_checkpoint)
        else: self.model.load_state_dict(merged_checkpoint)



class roberta_merged_model(nn.Module):
    def __init__(self, model_list: List, single_task:bool):
        super(roberta_merged_model, self).__init__()
        self.single_task = single_task
        self.classifiers = nn.ModuleList() 
        self.model_list_1 = copy.deepcopy(model_list)
        self.model_list_2 = copy.deepcopy(model_list)
        if(single_task == False):
            self.model_encoder = self.model_list_1[0].model.roberta
            # from /home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/roberta/modeling_roberta.py #line1329
            for i in range(len(model_list)):
                task_layers = nn.ModuleDict({
                    'fc': self.model_list_2[i].model.classifier  # 分类头
                })
                self.classifiers.append(task_layers)
        else:
            self.model = copy.deepcopy(model_list[0])

    def forward(self, input_ids, attention_mask):
        if(self.single_task == False):
            # from /home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/bert/modeling_bert.py #line 1680
            outputs = self.model_encoder(input_ids, attention_mask)
            sequence_output = outputs[0]
            outputs = []
            for task_layers in self.classifiers:
                class_output = task_layers['fc'](sequence_output)
                outputs.append(class_output)
            return outputs
        else: return self.model(input_ids, attention_mask)
    
    def load_weight(self, merged_checkpoint:dict):
        if(self.single_task == False):
            self.model_encoder.load_state_dict(merged_checkpoint)
        else: self.model.load_state_dict(merged_checkpoint)


class gpt2_merged_model(nn.Module):
    def __init__(self, model_list: List, single_task:bool, pad_token_id):
        super(gpt2_merged_model, self).__init__()
        self.config_pad_token_id = pad_token_id
        self.single_task = single_task
        self.classifiers = nn.ModuleList() 
        self.model_list_1 = copy.deepcopy(model_list)
        self.model_list_2 = copy.deepcopy(model_list)
        if(single_task == False):
            self.model_encoder = self.model_list_1[0].model.transformer
            # from/home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/gpt2/modeling_gpt2.py #1598
            for i in range(len(model_list)):
                task_layers = nn.ModuleDict({
                    'fc': self.model_list_2[i].model.score  # 分类头
                })
                self.classifiers.append(task_layers)
        else:
            self.model = copy.deepcopy(model_list[0])

    def forward(self, input_ids, attention_mask):
        # from/home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/gpt2/modeling_gpt2.py #1598
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        
        if input_ids is not None:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(input_ids, self.config_pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(self.model_encoder.device)

        if(self.single_task == False):
            # from/home/whj/miniconda3/envs/0407/lib/python3.11/site-packages/transformers/models/gpt2/modeling_gpt2.py #1598
            transformer_outputs = self.model_encoder(input_ids, attention_mask = attention_mask)
            hidden_states = transformer_outputs[0]
            outputs = []
            for task_layers in self.classifiers:
                logits = task_layers['fc'](hidden_states)
                batch_size, _ = input_ids.shape[:2]
                pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
                outputs.append(pooled_logits)
            return outputs
        else: return self.model(input_ids, attention_mask)
    
    def load_weight(self, merged_checkpoint:dict):
        if(self.single_task == False):
            self.model_encoder.load_state_dict(merged_checkpoint)
        else: self.model.load_state_dict(merged_checkpoint)