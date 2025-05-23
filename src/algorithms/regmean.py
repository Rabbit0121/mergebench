# regmean
# ref: https://github.com/jzhang538/BadMerging/blob/main/src/regmean.py
# ref: https://github.com/bloomberg/dataless-model-merging/tree/main



import copy
import re
import torch
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

from data.data_utils import get_loader, get_loader_nlp


class RegMean():
    def __init__(self, source, model_list, dataset_list, num_train_batch, processor, tokenizer):
        self.model_list = model_list
        self.dataset_list = dataset_list
        self.processor = processor
        self.tokenizer = tokenizer
        self.num_train_batch = num_train_batch
        self.source = source

    
    def get_merged_checkpoint(self):
        gram_list = []  
        with torch.no_grad():
            for model, dataset in zip(self.model_list, self.dataset_list):
                gram = self.compute_gram(model, dataset)
                gram_list.append(gram)

        regmean_avg_params = self.avg_merge(self.model_list, regmean_grams=gram_list)
        # print(type(regmean_avg_params))
        # import pdb; pdb.set_trace()
        merged_checkpoint = self.copy_params_to_checkpoint(regmean_avg_params, self.model_list[0])
        return merged_checkpoint

    def copy_params_to_checkpoint(self, avg_params, model):
        if(self.source == 'hf_vit'):
            merged_checkpoint = copy.deepcopy(model.model.vit.state_dict())
            paras_names = [name for name, _ in model.model.vit.named_parameters()]
        
        if(self.source == 'hf_clip'):
            merged_checkpoint = copy.deepcopy(model.model.vision_model.state_dict())
            paras_names = [name for name, _ in model.model.vision_model.named_parameters()]

        if(self.source == 'hf_bert'):
            merged_checkpoint = copy.deepcopy(model.model.bert.state_dict())
            paras_names = [name for name, _ in model.model.bert.named_parameters()]
        
        if(self.source == 'hf_roberta'):
            merged_checkpoint = copy.deepcopy(model.model.roberta.state_dict())
            paras_names = [name for name, _ in model.model.roberta.named_parameters()]
        if(self.source == 'hf_gpt2'):
            merged_checkpoint = copy.deepcopy(model.model.transformer.state_dict())
            paras_names = [name for name, _ in model.model.transformer.named_parameters()]

        


        for name in paras_names:
            merged_checkpoint[name].copy_(avg_params[name])
        return merged_checkpoint

    def filter_modules_by_regex(self, base_module, include_patterns, include_type):
        modules = {}
        for name, module in base_module.named_modules():
            valid_name = not include_patterns or any([re.match(patt, name) for patt in include_patterns])
            valid_type = not include_type or any([isinstance(module, md_cls) for md_cls in include_type])
            if valid_type and valid_name:
                modules[name] = module
        return modules
    
    
    def compute_gram(self, model, dataset):
        grams = {} 
        xn = {}

        def get_gram(name):
            def hook(module, input, output):
                x = input[0].detach()  # $[b,t,h]
                x = x.view(-1, x.size(-1))
                xtx = torch.matmul(x.transpose(0, 1), x)  # [h,h]
                if name not in grams:
                    grams[name] = xtx / x.size(0)
                    xn[name] = x.size(0)
                else:
                    grams[name] = (grams[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                    xn[name] += x.size(0)

            return hook
        if(self.source == 'hf_vit'):
            self.linear_modules = self.filter_modules_by_regex(model.model.vit, None, [torch.nn.Linear])
        if(self.source == 'hf_clip'):
            self.linear_modules = self.filter_modules_by_regex(model.model.vision_model, None, [torch.nn.Linear])
        if(self.source == 'hf_bert'):   
            self.linear_modules = self.filter_modules_by_regex(model.model.bert, None, [torch.nn.Linear])
        if(self.source == 'hf_roberta'):
            self.linear_modules = self.filter_modules_by_regex(model.model.roberta, None, [torch.nn.Linear])
            # print(self.linear_modules)
            # import pdb; pdb.set_trace()
        if(self.source == 'hf_gpt2'):
            self.linear_modules = self.filter_modules_by_regex(model.model.transformer, None, [Conv1D])
            # print(self.linear_modules)
            # import pdb; pdb.set_trace()
            # print(model.model.transformer.state_dict().keys())
            # import pdb; pdb.set_trace()


        # self.linear_modules = self.filter_modules_by_regex(model.model.vision_model, None, [torch.nn.Linear])
        # print("linear_modules", linear_modules)
        handles = []
        for name, module in self.linear_modules.items():
            handle = module.register_forward_hook(get_gram(name))
            handles.append(handle)
        if(self.source in ['hf_vit', 'hf_clip']):
        # get_loader(datasetname, batchsize, percent, num_workers, nonrmalize, processor/tokenizer)
            train_loader, _, _ = get_loader(dataset, 4, 1, 4, False, self.processor)
            # train_loader, _, _ = get_loader_nlp(dataset, 4, 0.01, 4, self.tokenizer)

            for i, batch in enumerate(tqdm(train_loader, total=self.num_train_batch, desc="computing gram")):
                if i >= self.num_train_batch:
                    break  
                # cv
                inputs, _ = batch
                # nlp
                # input_ids, attetion mask
                inputs = inputs.cuda()
                model(inputs)
        if(self.source in ['hf_bert', 'hf_roberta', 'hf_gpt2']):
            if(self.source == 'hf_gpt2'):
                model.model.config.pad_token_id = self.tokenizer.eos_token_id
            train_loader, _, _ = get_loader_nlp(dataset, 4, 1, 4, self.tokenizer)
            for i, batch in enumerate(tqdm(train_loader, total=self.num_train_batch, desc="computing gram")):
                if i >= self.num_train_batch:
                    break
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()

                model(input_ids, attention_mask)      

        for handle in handles:
            handle.remove()

        return grams


    def reduce_non_diag(self, cov_mat, a):
        diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
        non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
        weight = diag_weight + non_diag_weight
        ret = cov_mat * weight
        return ret
    
    def regmean_merge(self, all_params, all_grams):
        avg_params = {}
        cnt = 0

        for name in all_params:
            h_avged = False
            name1 = name
            
            if name1.endswith('.weight'):
                module_name = name1[:-len('.weight')]
                if module_name in all_grams[0]:
                    # print(module_name)
                    cnt += 1
                    gram_m_ws, grams = [], []

                    for model_id, model_grams in enumerate(all_grams):
                        param_grams = model_grams[module_name]
                        param_grams = self.reduce_non_diag(param_grams, a=0.1)

                        param = all_params[name][model_id]
                        if self.source == 'hf_gpt2':
                            param = param.transpose(0, 1)
                        gram_m_ws.append(torch.matmul(param_grams, param.transpose(0, 1)))
                        grams.append(param_grams)
                    sum_gram = sum(grams)
                    sum_gram_m_ws = sum(gram_m_ws)
                    # 保证可逆
                    epsilon = 1e-10
                    sum_gram = sum_gram + epsilon * torch.eye(sum_gram.size(0), device=sum_gram.device)
                    sum_gram_inv = torch.inverse(sum_gram)
                    wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)
                    w = wt.transpose(0, 1)
                    if self.source == 'hf_gpt2':
                        w = w.transpose(0, 1)
                    avg_params[name] = w
                    h_avged = True
            
            if not h_avged:  # if not averaged with regmean, then do simple avg
                # import pdb; pdb.set_trace()
                avg_params[name] = torch.stack(all_params[name], 0).mean(0)

        print(cnt, len(all_grams[0]))
        return avg_params
    
    def avg_merge(self, local_models, regmean_grams=None, **kwargs): # **kwargs?
        params = {}
        for local_model in local_models:
            # fix to match vit,clip,nlp
            if(self.source == 'hf_vit'):
                n2p = {k: v for k, v in local_model.model.vit.named_parameters()}
                merge_param_names = [name for name, _ in local_model.model.vit.named_parameters()]
  
            if(self.source == 'hf_clip'):
                n2p = {k: v for k, v in local_model.model.vision_model.named_parameters()}
                merge_param_names = [name for name, _ in local_model.model.vision_model.named_parameters()]
            
            if(self.source == 'hf_bert'):
                n2p = {k: v for k, v in local_model.model.bert.named_parameters()}
                merge_param_names = [name for name, _ in local_model.model.bert.named_parameters()]
            
            if(self.source == 'hf_roberta'):
                n2p = {k: v for k, v in local_model.model.roberta.named_parameters()}
                merge_param_names = [name for name, _ in local_model.model.roberta.named_parameters()]
            if(self.source == 'hf_gpt2'):
                n2p = {k: v for k, v in local_model.model.transformer.named_parameters()}
                merge_param_names = [name for name, _ in local_model.model.transformer.named_parameters()]

            for n in merge_param_names:
                if n not in params:
                    params[n] = []
                params[n].append(n2p[n])

        if regmean_grams:  # regmean average
            # print(len(regmean_grams[0]))
            # import pdb; pdb.set_trace()
            avg_params = self.regmean_merge(params, regmean_grams)

        else:  # simple average
            avg_params = {k: torch.stack(v, 0).mean(0) for k, v in params.items()}

        return avg_params