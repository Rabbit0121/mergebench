import argparse
import copy
import sys
from typing import List, Type
from merge_class import simclr_merged_model
import torch

sys.path.append('..')
sys.path.append('../..')
from src.merge.merge_utils_1 import combine_loader
from data.data_utils import get_loader, get_num_classes
from utils.files import get_hyperparameters, write_to_csv
from merge_utils import are_elements_equal, evaluate, evaluate_upstream_model, reset_bn_stats
from algorithms.soups import ModelSoups
from algorithms.task_arithmetic import TaskVector
from algorithms.ties import TIES
from algorithms.dare import DARE
from algorithms.slerp import SLERP
from algorithms.adamerging import AdaMerging_Layer, AdaMerging_Task
from algorithms.adamerging_nlp import AdaMerging_Layer_NLP, AdaMerging_Task_NLP
from algorithms.regmean import RegMean
from models.simclr.ft_class import ImageClassifier
from merge_cnn_single import r50x1_model_path_list,r101x1_model_path_list,  r152x1_model_path_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [r50x1_model_path_list,  r101x1_model_path_list, r152x1_model_path_list]
type_list = ['r50_1x', 'r101_1x',  'r152_1x']
datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']

for model_path_list, model_type in zip(path_list, type_list):
    acc_upstream_dict = {}
    for i in range(len(model_path_list)):
        for dataset in model_path_list[i].keys():
            for j in range(len(model_path_list[i][dataset])):
                acc_upstream_dict[str(dataset) + '_' + str(j+1)] = evaluate_upstream_model(ImageClassifier.load_from_checkpoint(model_path_list[i][dataset][j]), dataset, 32, 4, False)

    for i in range(len(model_path_list)):
        for dataset in model_path_list[i].keys():
            model_list = []  # 每次重新初始化
            ckpt_list = []
            merged_model = torch.load(f'/data4/xxx/0407/finetune/simclr/{dataset}/{model_type}_sk0_pretrain.pth')
            pre_model = torch.load(f'/data4/xxx/0407/finetune/simclr/{dataset}/{model_type}_sk0_pretrain.pth')
            for j in range(len(model_path_list[i][dataset])):
                model_path = model_path_list[i][dataset][j]
                model = ImageClassifier.load_from_checkpoint(model_path)
                model_list.append(model)
                ckpt_list.append(DARE(TaskVector(pre_model.state_dict(), model.state_dict()), mask_rate = 0.2, use_rescale = True, mask_strategy = 'magnitude'))
                # ckpt_list.append(TaskVector(pre_model.state_dict(), model.state_dict()))
            # task vector
            for coef in range(8,15,1):
                coef = coef / 10
                merged_ckpt = TIES(ckpt_list, mask_rate=0.2,merge_func="mean", pre = pre_model.state_dict(),  scaling_coef=coef)
                merged_model = torch.load(f'/data4/xxx/0407/finetune/simclr/{dataset}/{model_type}_sk0_pretrain.pth')
                merged_model.load_state_dict(merged_ckpt)
                # evaluate
                acc_merged = evaluate_upstream_model(merged_model, dataset, batch_size=32, num_workers=4, normalize=False)
                acc_upstream = [ acc_upstream_dict[str(dataset) + '_' + str(1)], acc_upstream_dict[str(dataset) + '_' + str(2)] ]
                print(acc_upstream_dict, acc_merged)
                headers=["datasets", "acc_upstream", "acc_merged"]
                data={"datasets":model_type + dataset, "acc_upstream":acc_upstream, "acc_merged":acc_merged}
                result_path = '../results/new_merge_2_resetbn/simclr/single_tiesdare.csv'
                write_to_csv(result_path, headers, data)
                merged_model = reset_bn_stats(merged_model, get_loader(dataset, 32, 4, False)[0], device)
                acc_merged = evaluate_upstream_model(merged_model, dataset, batch_size=32, num_workers=4, normalize=False)
                print(acc_upstream_dict, acc_merged)
                headers=["datasets", "acc_upstream", "acc_merged"]
                data={"datasets":model_type + dataset, "acc_upstream":acc_upstream, "acc_merged":acc_merged}
                result_path = '../results/new_merge_2_resetbn/simclr/single_tiesadre.csv'
                write_to_csv(result_path, headers, data)
                # import pdb; pdb.set_trace()