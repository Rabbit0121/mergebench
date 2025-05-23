import argparse
import copy
import sys
from typing import List, Type
from merge_class import simclr_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from src.merge.merge_utils_1 import combine_loader
from utils.files import get_hyperparameters, write_to_csv
from merge_utils_1 import are_elements_equal, evaluate, evaluate_upstream_model, reset_bn_stats
from algorithms.soups import ModelSoups
from models.simclr.ft_class import ImageClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path_list = [
                '/data4/xxx/0407/finetune/simclr/gtsrb/r50_1x_sk0/version_3/checkpoints/epoch=09_step=6660_val_loss=0.146038.ckpt',
                  '/data4/xxx/0407/finetune/simclr/cifar10/r50_1x_sk0/version_1/checkpoints/epoch=19_step=25000_val_loss=0.140188.ckpt',              
                   '/data4/xxx/0407/finetune/simclr/svhn/r50_1x_sk0/version_2/checkpoints/epoch=12_step=23816_val_loss=0.175431.ckpt',
                  '/data4/xxx/0407/finetune/simclr/sun397/r50_1x_sk0/version_7/checkpoints/epoch=11_step=12240_val_loss=1.428785.ckpt',
                #   '/data4/xxx/0407/finetune/simclr/dtd/r50_1x_sk0/version_2/checkpoints/epoch=08_step=531_val_loss=1.448840.ckpt',
                    '/data4/xxx/0407/finetune/simclr/dtd/r50_1x_sk0/version_7/checkpoints/epoch=16_step=510_val_loss=1.339717.ckpt',
                #   '/data4/xxx/0407/finetune/simclr/cifar100/r50_1x_sk0/version_4/checkpoints/epoch=18_step=23750_val_loss=1.438568.ckpt',
                    '/data4/xxx/0407/finetune/simclr/cifar100/r50_1x_sk0/version_1/checkpoints/epoch=17_step=22500_val_loss=0.783409.ckpt',
                  '/data4/xxx/0407/finetune/simclr/stanfordcars/r50_1x_sk0/version_8/checkpoints/epoch=16_step=1734_val_loss=0.917776.ckpt',
                #   '/data4/xxx/0407/finetune/simclr/eurosat/r50_1x_sk0/version_9/checkpoints/epoch=11_step=3048_val_loss=0.302482.ckpt',
                '/data4/xxx/0407/finetune/simclr/eurosat/r50_1x_sk0/version_1/checkpoints/epoch=18_step=9633_val_loss=0.058038.ckpt',
                  '/data4/xxx/0407/finetune/simclr/mnist/r50_1x_sk0/version_0/checkpoints/epoch=19_step=30000_val_loss=0.062541.ckpt'
                  ] 
datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']
acc_upstream_dict = {}
for i in range(len(model_path_list)):
    acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model(ImageClassifier.load_from_checkpoint(model_path_list[i]), datasets_list[i], 32, 4, False, None)
model_type = 'r50_1x'

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
        # soups
        soups = ModelSoups(ckpt_list)
        merge_ckpt = soups.run()
        merged_model = simclr_merged_model(model_list, False)
        merged_model.load_weight(merge_ckpt)
        # evaluate
        acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
        merged_model = reset_bn_stats(merged_model, combine_loader(datasets_list[i], datasets_list[j]), device)
        _, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False)
        print(acc_upstream, acc_merged)
        headers=["datasets", "acc_upstream", "acc_merged"]
        data={"datasets":model_type + str(dataset_list), "acc_upstream":acc_upstream, "acc_merged":acc_merged}
        result_path = '../results/new_merge_2_resetbn/simclr/soups.csv'
        write_to_csv(result_path, headers, data)