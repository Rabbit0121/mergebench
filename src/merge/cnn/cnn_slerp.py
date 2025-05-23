import argparse
import copy
import sys
from typing import List, Type
from merge_class import simclr_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from utils.files import get_hyperparameters, write_to_csv
from merge_utils import are_elements_equal, evaluate, evaluate_upstream_model
from algorithms.slerp import SLERP
from models.simclr.ft_class import ImageClassifier
from merge_cnn import r50x1_model_path_list, r50x2_model_path_list, r101x1_model_path_list, r101x2_model_path_list, r152x1_model_path_list, r152x2_model_path_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [r50x1_model_path_list, r50x2_model_path_list, r101x1_model_path_list, r101x2_model_path_list, r152x1_model_path_list, r152x2_model_path_list]
type_list = ['r50_1x', 'r50_2x', 'r101_1x', 'r101_2x', 'r152_1x', 'r152_2x']
datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']

for model_path_list, model_type in zip(path_list, type_list):
    acc_upstream_dict = {}
    for i in range(len(model_path_list)):
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model(ImageClassifier.load_from_checkpoint(model_path_list[i]), datasets_list[i], 32, 4, False)
    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            model_list = []  
            ckpt_list = []
            dataset_list = []
            dataset_list.append(datasets_list[i])
            dataset_list.append(datasets_list[j])
            model1 = ImageClassifier.load_from_checkpoint(model_path_list[i])
            model2 = ImageClassifier.load_from_checkpoint(model_path_list[j])
            model_list.append(model1)
            model_list.append(model2)
            ckpt_list.append(model1.model.net.state_dict())
            ckpt_list.append(model2.model.net.state_dict())
            # slerp
            for degree in range(1, 10, 1):
                degree = degree / 10
                merge_ckpt = SLERP(ckpt_list[0], ckpt_list[1], degree)
                merged_model = simclr_merged_model(model_list, False)
                merged_model.load_weight(merge_ckpt)
                # evaluate
                reset_bn_stats(merged_model, combine_loader_multi(dataset_list), device)
                acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
                _, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False)
                print(acc_upstream, acc_merged)
                headers=["datasets", "degree", "acc_upstream", "acc_merged"]
                data={"datasets":model_type + str(dataset_list), "acc_upstream":acc_upstream, "acc_merged":acc_merged, "degree":degree}
                result_path = '../results/new_merge_2/simclr/slerp.csv'
                write_to_csv(result_path, headers, data)