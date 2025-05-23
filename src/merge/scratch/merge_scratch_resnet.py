# models: resnet-18, 34, 50, 101, 152
# algorithms: model soups, slerp, fisher, regmean(不适用)
# sources: pytorch mmpretrain
# dataset: imagenet

import copy
import torch

# load model, load dataset, verify dataset processing, previous acc, merged acc, save
import torchvision.models as models
import torch
import sys

import torch
sys.path.append('..')
sys.path.append('../..')
from data.data_utils import get_loader
from utils.files import write_to_csv
from merge_utils import evaluate_upstream_model, transform, reset_bn_stats
from algorithms.soups import ModelSoups
from algorithms.slerp import SLERP
import torch
import sys

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_names = [
               "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", 
               "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
               # "densenet121", "densenet161", "densenet169", "densenet201"
               ]
model_creators = [getattr(models, name) for name in model_names]
# pytorch
models_pytorch_list = [creator().to(device) for creator in model_creators]
for model, name in zip(models_pytorch_list, model_names):
    model.load_state_dict(torch.load(f'/data4/xxx/0407/models/scratch/pytorch/imagenet/{name}.pth').state_dict())
# print(models_pytorch_list[0].state_dict().keys())

# mmpretrain
models_mmpretrain_list = [creator().to(device) for creator in model_creators]
for model, name in zip(models_mmpretrain_list, model_names):
    checkpoint = torch.load(f'/data4/xxx/0407/models/scratch/mmpretrain/imagenet/{name}.pth')
    new_checkpoint = transform(checkpoint, name)
    model.load_state_dict(new_checkpoint)
# print(models_mmpretrain_list[0].state_dict().keys())

# merge: single_task

# for model_pytorch, model_mmpretrain, name in zip(models_pytorch_list, models_mmpretrain_list, model_names):
#     model_list = [model_pytorch, model_mmpretrain]
#     ckpt_list = [model_pytorch.state_dict(), model_mmpretrain.state_dict()]
#     sources_list = ['pytorch', 'mmpretrain']
#     # evaluate upstream
#     acc_upstream = []
#     for model, sources in zip(model_list, sources_list):
#         acc_upstream.append(evaluate_upstream_model(model, 'imagenet', 32, 8, normalize=True, processor=None))
#     print(acc_upstream)
#     # soups
#     soups = ModelSoups(ckpt_list)
#     merged_ckpt = soups.run()
#     merged_model = copy.deepcopy(model_pytorch)
#     merged_model.load_state_dict(merged_ckpt)
#     reset_bn_stats(merged_model, get_loader('imagenet', 32, 8, normalize=True, processor=None)[0], device)
#     # evaluate
#     acc_merged = evaluate_upstream_model(merged_model, 'imagenet', 32, 8, normalize=True, processor=None)
#     print(acc_merged)
#     headers=["sources", "model", "acc_upstream", "acc_merged"]
#     data={"model":name, "acc_upstream":acc_upstream, "acc_merged":acc_merged, "sources":sources_list}
#     result_path = '../results/new_merge_2/scratch/single_soups.csv'
#     # import pdb; pdb.set_trace()
#     write_to_csv(result_path, headers, data)

#     # slerp
#     for degree in range(1, 10, 1):
#         degree = degree / 10
#         merged_ckpt = SLERP(ckpt_list[0], ckpt_list[1], degree)
#         mmerged_model = copy.deepcopy(model_pytorch)
#         merged_model.load_state_dict(merged_ckpt)
#         # evaluate
#         reset_bn_stats(merged_model, get_loader('imagenet', 32, 8, normalize=True, processor=None)[0], device)
#         acc_merged = evaluate_upstream_model(merged_model, 'imagenet', 32, 8, normalize=True, processor=None)
#         headers=["sources", "model", "degree", "acc_upstream", "acc_merged"]
#         data={"model":name, "acc_upstream":acc_upstream, "acc_merged":acc_merged, "sources":sources_list, "degree":degree}
#         result_path = '../results/new_merge_2/scratch/single_slerp.csv'
#         write_to_csv(result_path, headers, data)


# densenet 121 
# pytorch timm(hf)
model_pytorch = models.densenet121(weights=None).to(device)
model_pytorch.load_state_dict(torch.load(f'/data4/xxx/0407/models/scratch/pytorch/imagenet/densenet121.pth').state_dict())

model_timm = models.densenet121(weights=None).to(device)
model_timm.load_state_dict(torch.load('/data4/xxx/0407/models/scratch/timm/imagenet/densenet121.bin'))

acc_upstream = [evaluate_upstream_model(model_pytorch, 'imagenet', 32, 8, normalize=True, processor=None),
                evaluate_upstream_model(model_timm, 'imagenet', 32, 8, normalize=True, processor=None)]
print(acc_upstream)
ckpt_list = [model_pytorch.state_dict(), model_timm.state_dict()]
sources_list = ['pytorch', 'timm']
# soups
soups = ModelSoups(ckpt_list)
merged_ckpt = soups.run()
merged_model = copy.deepcopy(model_pytorch)
merged_model.load_state_dict(merged_ckpt)
# evaluate
reset_bn_stats(merged_model, get_loader('imagenet', 32, 8, normalize=True, processor=None)[0], device)
acc_merged = evaluate_upstream_model(merged_model, 'imagenet', 32, 8, normalize=True, processor=None)
headers=["sources", "model", "acc_upstream", "acc_merged"]
data={"model":'densenet121', "acc_upstream":acc_upstream, "acc_merged":acc_merged, "sources":sources_list}
result_path = '../results/new_merge_2/scratch/single_soups.csv'
write_to_csv(result_path, headers, data)

# slerp
for degree in range(1, 10, 1):
    degree = degree / 10
    merged_ckpt = SLERP(ckpt_list[0], ckpt_list[1], degree)
    mmerged_model = copy.deepcopy(model_pytorch)
    merged_model.load_state_dict(merged_ckpt)
    # evaluate
    reset_bn_stats(merged_model, get_loader('imagenet', 32, 8, normalize=True, processor=None)[0], device)
    acc_merged = evaluate_upstream_model(merged_model, 'imagenet', 32, 8, normalize=True, processor=None)
    headers=["sources", "model", "degree", "acc_upstream", "acc_merged"]
    data={"model":'densenet', "acc_upstream":acc_upstream, "acc_merged":acc_merged, "sources":sources_list, "degree":degree}
    result_path = '../results/new_merge_2/scratch/single_slerp.csv'
    write_to_csv(result_path, headers, data)
