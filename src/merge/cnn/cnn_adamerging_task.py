import argparse
import copy
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import sys
from typing import List, Type
from merge_class import simclr_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from src.data.data_utils import get_loader
from utils.files import get_hyperparameters, write_to_csv
from merge_utils_1 import are_elements_equal, combine_loader_multi, evaluate, evaluate_upstream_model, reset_bn_stats
from algorithms.adamerging import AdaMerging_Layer, AdaMerging_Task
from algorithms.task_arithmetic import TaskVector
from models.simclr.ft_class import ImageClassifier
from merge_cnn import r50x1_model_path_list, r50x2_model_path_list, r101x1_model_path_list, r101x2_model_path_list, r152x1_model_path_list, r152x2_model_path_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [r50x1_model_path_list, r50x2_model_path_list, r101x1_model_path_list, r101x2_model_path_list, r152x1_model_path_list, r152x2_model_path_list]
type_list = ['r50_1x', 'r50_2x', 'r101_1x', 'r101_2x', 'r152_1x', 'r152_2x']
datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']
pre_model_path_list = [  '/data4/xxx/0407/finetune/simclr/gtsrb/r50_1x_sk0_pretrain.pth',
                    '/data4/xxx/0407/finetune/simclr/gtsrb/r50_2x_sk0_pretrain.pth',
                    '/data4/xxx/0407/finetune/simclr/gtsrb/r101_1x_sk0_pretrain.pth',
                    '/data4/xxx/0407/finetune/simclr/gtsrb/r101_2x_sk0_pretrain.pth',
                    '/data4/xxx/0407/finetune/simclr/gtsrb/r152_1x_sk0_pretrain.pth',
                    '/data4/xxx/0407/finetune/simclr/gtsrb/r152_2x_sk0_pretrain.pth']
test_loaders_list = [get_loader(dataset, 32, 1, 4, False, None)[2] for dataset in datasets_list]
for model_path_list, model_type, pre_model_path in zip(path_list, type_list, pre_model_path_list):
    acc_upstream_dict = {}
    for i in range(len(model_path_list)):
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model(ImageClassifier.load_from_checkpoint(model_path_list[i]), datasets_list[i], 32, 4, False)
        # print(acc_upstream_dict)
    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            # adamerging-task
            for max_steps in [100, 300, 500, 700, 900, 1000]:
                if(max_steps != 300):
                    continue
                max_steps=1
                model_list = []  
                ckpt_list = []
                dataset_list = []
                test_loader_list = []
                dataset_list.append(datasets_list[i])
                dataset_list.append(datasets_list[j])
                test_loader_list.append(test_loaders_list[i])
                test_loader_list.append(test_loaders_list[j])
                model1 = ImageClassifier.load_from_checkpoint(model_path_list[i])
                model2 = ImageClassifier.load_from_checkpoint(model_path_list[j])
                model_list.append(model1)
                model_list.append(model2)
                pre_model = torch.load(pre_model_path)
                ckpt_list.append(TaskVector(pre_model.model.net.state_dict(), model1.model.net.state_dict()))
                ckpt_list.append(TaskVector(pre_model.model.net.state_dict(), model2.model.net.state_dict()))
                adamerging_task = AdaMerging_Task('simclr', model_list, pre_model.model.net, ckpt_list, 1e-3, test_loader_list)
                name = f'{dataset_list[0]}_{dataset_list[1]}'
                logger = TensorBoardLogger(f'/data4/xxx/0407/adamerging_task/simclr/{model_type}', name = name)
                trainer = pl.Trainer(max_steps= max_steps, devices=1, accelerator="gpu", deterministic=True, logger=logger)
                trainer.fit(adamerging_task)
                merged_ckpt = adamerging_task.get_merged_ckpt()
                merged_coff = adamerging_task.last_step_lambda
                merged_loss = adamerging_task.last_step_loss
                merged_model = simclr_merged_model(model_list, False)
                merged_model.load_weight(merged_ckpt)
                # evaluate
                acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
                reset_bn_stats(merged_model, combine_loader_multi(dataset_list), device)
                _, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False)
                print(acc_upstream, acc_merged)
                headers=["datasets", "type", "max_steps", "merged_coff", "merged_loss", "acc_upstream", "acc_merged"]
                data={"datasets":model_type + f'_{dataset_list[0]}_{dataset_list[1]}', "acc_upstream":acc_upstream, "acc_merged":acc_merged, 
                      "type":"adamerging-task", "max_steps":max_steps, "merged_coff":merged_coff, "merged_loss":merged_loss}
                result_path = '../results/new_merge_2_appendix/simclr/adamerging_task.csv'
                write_to_csv(result_path, headers, data)
                # import pdb; pdb.set_trace()
                