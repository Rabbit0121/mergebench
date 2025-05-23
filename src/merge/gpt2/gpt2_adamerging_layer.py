import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import sys
from merge_class import gpt2_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from src.algorithms.ties import get_ties_ckpt
from src.data.data_utils import get_loader_nlp
from utils.files import write_to_csv
from merge_utils_1 import evaluate_nlp, evaluate_upstream_model_nlp
from algorithms.adamerging_nlp import AdaMerging_Task_NLP, AdaMerging_Layer_NLP
from algorithms.task_arithmetic import TaskVector
from ft_class import GPT2SequenceClassifier
from merge_gpt2 import gpt2
from transformers import GPT2Tokenizer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [gpt2]
type_list = ['gpt2']
tokenizer_list  = ['/data4/xxx/0407/models/hf_gpt2/gpt2']
pre_model_path_list = ["/data4/xxx/0407/ftnew/hf_gpt2/cola/gpt2_pretrain.pth"]
datasets_list = ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte']

for model_path_list, model_type, model_tokenizer_path, pre_model_path in zip(path_list, type_list, tokenizer_list, pre_model_path_list):
    tokenizer = GPT2Tokenizer.from_pretrained(model_tokenizer_path + '/tokenizer')
    tokenizer.pad_token = tokenizer.eos_token
    test_loaders_list = [get_loader_nlp(dataset, 8, 1, 4, tokenizer)[0] for dataset in datasets_list]
    acc_upstream_dict = {}
    model_path_list = model_path_list[:2]
    for i in range(len(model_path_list)):
        model = GPT2SequenceClassifier.load_from_checkpoint(model_path_list[i])
        model.model.config.pad_token_id = tokenizer.eos_token_id
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model_nlp(model, datasets_list[i], 32, 4, tokenizer)
    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            # adamerging-task
            for max_steps in [100, 300, 500, 700, 900, 1000]:
                if(max_steps != 500):
                    continue
                pre_model = torch.load(pre_model_path)
                model_list = [] 
                ckpt_list = []
                dataset_list = []
                test_loader_list = []
                dataset_list.append(datasets_list[i])
                dataset_list.append(datasets_list[j])
                test_loader_list.append(test_loaders_list[i])
                test_loader_list.append(test_loaders_list[j])
                model1 = GPT2SequenceClassifier.load_from_checkpoint(model_path_list[i])
                model2 = GPT2SequenceClassifier.load_from_checkpoint(model_path_list[j])
                model_list.append(model1)
                model_list.append(model2)
                ckpt_list.append(TaskVector(pre_model.model.transformer.state_dict(), model1.model.transformer.state_dict()))
                ckpt_list.append(TaskVector(pre_model.model.transformer.state_dict(), model2.model.transformer.state_dict()))
                # plus version
                ties_ckpt_list = get_ties_ckpt(ckpt_list, pre_model.model.transformer.state_dict())
                adamerging_task = AdaMerging_Layer_NLP('hf_gpt2', model_list, pre_model.model.transformer, ties_ckpt_list, 1e-3, test_loader_list, tokenizer.pad_token_id)
                # adamerging_task = AdaMerging_Layer_NLP('hf_gpt2', model_list, pre_model.model.transformer, ckpt_list, 1e-3, test_loader_list, tokenizer.pad_token_id)
                name = f'{dataset_list[0]}_{dataset_list[1]}'
                logger = TensorBoardLogger(f'/data4/xxx/0407/adamerging_layer++/hf_gpt2/{model_type}', name = name)
                trainer = pl.Trainer(max_steps= max_steps, devices=1, accelerator="gpu", deterministic=True, logger=logger)
                trainer.fit(adamerging_task)
                merged_ckpt = adamerging_task.get_merged_ckpt()
                merged_coff = adamerging_task.last_step_lambda
                merged_loss = adamerging_task.last_step_loss
                merged_model = gpt2_merged_model(model_list, False, tokenizer.eos_token_id)
                merged_model.load_weight(merged_ckpt)
                # evaluate
                acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
                _, acc_merged = evaluate_nlp(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, tokenizer=tokenizer)
                print(acc_upstream, acc_merged)
                headers=["datasets", "type", "max_steps", "merged_loss", "acc_f1_mcc_upstream", "acc_f1_mcc_merged"]
                data={"datasets":model_type + f'_{dataset_list[0]}_{dataset_list[1]}', "acc_f1_mcc_upstream":acc_upstream, "acc_f1_mcc_merged":acc_merged, 
                        "type":"adamerging-layer++", "max_steps":max_steps, "merged_loss":merged_loss}
                result_path = '../results/new_merge_2/gpt2/adamerging_layer++.csv'
                write_to_csv(result_path, headers, data)
                # import pdb; pdb.set_trace()
                