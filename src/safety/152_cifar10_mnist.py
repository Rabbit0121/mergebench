# domain: cv
# model: resnet50, cvit-b/16, clip-b/16
# algorithm: soups, ta, tadare, ties, tiesdare, slerp, fisher, regmean, adamerging
# type: single, multi
# dataset: gtsrb, sun397

import copy
from pytorch_lightning import Trainer
import torch
import sys

import torch



sys.path.append('..')
sys.path.append('../..')

from algorithms.adamerging import AdaMerging_Layer
from algorithms.slerp import SLERP
from algorithms.ties import TIES
from algorithms.task_arithmetic import TaskVector
from merge.merge_utils_1 import combine_loader
from utils.files import write_to_csv
from merge.merge_class import simclr_merged_model
from merge.merge_utils import evaluate, evaluate_upstream_model, reset_bn_stats
from algorithms.soups import ModelSoups
from models.simclr.ft_class import ImageClassifier
from data.data_utils import get_loader
from autoattack import AutoAttack
from pgd_adv_train_class import LitPGDTrainer
from pytorch_lightning.loggers import TensorBoardLogger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




import torch
from itertools import cycle, islice


def evaluate_upstream_model_robsust(model, dataset, batch_size, num_workers, normalize):
    test_loader = get_loader(dataset, batch_size, num_workers, normalize)[1]
    print(len(test_loader.dataset))
    model.eval()
    model.to(device)
    def create_forward_pass():
        def forward_pass(x):
            x = model(x)
            return x
        return forward_pass
    
    forward_pass = create_forward_pass()
    l = [x for (x, _) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (_, y) in test_loader]
    y_test = torch.cat(l, 0)
    log_path = 'r50test.csv'
    norm = 'Linf'
    adversary = AutoAttack(forward_pass, norm = norm, eps = 8. / 255., log_path = log_path, verbose = True)
    # ckeanacc: all test
    cleanacc = adversary.clean_accuracy(x_test, y_test, bs=64)
    print(cleanacc)
    # robustacc: all test
    n_ex = 1000
    with torch.no_grad(): 
        # individual
        with open(log_path, "a") as f:
            adversary = AutoAttack(forward_pass, norm = norm, eps = 8. / 255., log_path = log_path, verbose = True, version='standard')
            _ = adversary.run_standard_evaluation(
                x_test[:n_ex], y_test[:n_ex], bs=64, return_labels=True)
    
def evaluate_merged_model_robsust(model, dataset, batch_size, num_workers, normalize, idx):
    test_loader = get_loader(dataset, batch_size, num_workers, normalize)[1]
    print(len(test_loader.dataset))
    model.eval()
    model.to(device)
    def create_forward_pass(idx):
        def forward_pass(x):
            x = model(x)
            return x[idx]
        return forward_pass
    
    forward_pass = create_forward_pass(idx)
    l = [x for (x, _) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (_, y) in test_loader]
    y_test = torch.cat(l, 0)
    log_path = '152_cifar10_mnist.csv'
    norm = 'Linf'
    adversary = AutoAttack(forward_pass, norm = norm, eps = 8. / 255., log_path = log_path, verbose = True)
    # ckeanacc: all test
    cleanacc = adversary.clean_accuracy(x_test, y_test, bs=64)
    print(cleanacc)
    # robustacc: all test
    n_ex = 1000
    with torch.no_grad(): 
        # individual
        with open(log_path, "a") as f:
            adversary = AutoAttack(forward_pass, norm = norm, eps = 8. / 255., log_path = log_path, verbose = True, version='standard')
            _ = adversary.run_standard_evaluation(
                x_test[:n_ex], y_test[:n_ex], bs=64, return_labels=True)
    return




# **************soups*******************
print("soups\n")
model_mnist = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/mnist/version_0/checkpoints/epoch=0-step=469-train_loss_adv=0.09.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/mnist/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_mnist, 'mnist', 32, 4, False))
# evaluate_upstream_model_robsust(model_mnist, 'mnist', 32, 4, False)

model_cifar10 = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/cifar10/version_0/checkpoints/epoch=103-step=40664-train_loss_adv=0.45.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/cifar10/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_mnist, 'cifar10', 32, 4, False))
# evaluate_upstream_model_robsust(model_mnist, 'cifar10', 32, 4, False)
model_list = [model_mnist.model, model_cifar10.model]
ckpt_list = [model_mnist.model.model.net.state_dict(), model_cifar10.model.model.net.state_dict()]

soups = ModelSoups(ckpt_list)
merged_ckpt = soups.run()
merged_model = simclr_merged_model(model_list, False)
merged_model.load_weight(merged_ckpt)
merged_model = reset_bn_stats(merged_model, combine_loader('mnist', 'cifar10'), device)
print(evaluate(model_list, merged_model, ['mnist', 'cifar10'], 32, 4, False))
print(evaluate_merged_model_robsust(merged_model, 'mnist', 32, 4, False, 0))
print(evaluate_merged_model_robsust(merged_model, 'cifar10', 32, 4, False, 1))



# *************ta*************
print("ta\n")

model_mnist = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/mnist/version_0/checkpoints/epoch=0-step=469-train_loss_adv=0.09.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/mnist/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_mnist, 'mnist', 32, 4, False))
# evaluate_upstream_model_robsust(model_mnist, 'mnist', 32, 4, False)
model_cifar10 = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/cifar10/version_0/checkpoints/epoch=103-step=40664-train_loss_adv=0.45.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/cifar10/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_cifar10, 'cifar10', 32, 4, False))
# evaluate_upstream_model_robsust(model_cifar10, 'cifar10', 32, 4, False)

pre_model = torch.load((f'/data4/xxx/0407/finetune/simclr/cifar10/r152_1x_sk0_pretrain.pth'))

model_list = [model_mnist.model, model_cifar10.model]
ckpt_list = [TaskVector(pre_model.model.net.state_dict(), model_mnist.model.model.net.state_dict()), 
             TaskVector(pre_model.model.net.state_dict(), model_cifar10.model.model.net.state_dict()), 
             ]
sum_tv = sum(ckpt_list)
for coeff in range(6, 7, 1):
    coeff = coeff / 10
    merged_ckpt = sum_tv.apply_to(pre_model.model.net.state_dict(), scaling_coef=coeff)
    merged_model = simclr_merged_model(model_list, False)
    merged_model.load_weight(merged_ckpt)
    merged_model = reset_bn_stats(merged_model, combine_loader('mnist', 'cifar10'), device)
    print(coeff, evaluate(model_list, merged_model, ['mnist', 'cifar10'], 32, 4, False))
    print(evaluate_merged_model_robsust(merged_model, 'mnist', 32, 4, False, 0))
    print(evaluate_merged_model_robsust(merged_model, 'cifar10', 32, 4, False, 1))

# import pdb; pdb.set_trace()
# ****************ties*****************
print("ties\n")
model_mnist = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/mnist/version_0/checkpoints/epoch=0-step=469-train_loss_adv=0.09.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/mnist/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_mnist, 'mnist', 32, 4, False))
# evaluate_upstream_model_robsust(model_mnist, 'mnist', 32, 4, False)
model_cifar10 = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/cifar10/version_0/checkpoints/epoch=103-step=40664-train_loss_adv=0.45.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/cifar10/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_cifar10, 'cifar10', 32, 4, False))
# evaluate_upstream_model_robsust(model_cifar10, 'cifar10', 32, 4, False)

pre_model = torch.load((f'/data4/xxx/0407/finetune/simclr/cifar10/r152_1x_sk0_pretrain.pth'))

model_list = [model_mnist.model, model_cifar10.model]
ckpt_list = [TaskVector(pre_model.model.net.state_dict(), model_mnist.model.model.net.state_dict()), 
             TaskVector(pre_model.model.net.state_dict(), model_cifar10.model.model.net.state_dict()), 
             ]
for ties_rate in range(1,10,1):
    # 为防止暴力搜索，ties rate指定20%
    if(ties_rate != 2):
        continue
    ties_rate = ties_rate /  10
    for coef in range(6,7,1):
        coef = coef / 10
        merge_ckpt = TIES(ckpt_list, mask_rate=ties_rate,merge_func="mean", pre = pre_model.model.net.state_dict(),  scaling_coef=coef)
        merged_model = simclr_merged_model(model_list, False)
        merged_model.load_weight(merge_ckpt)
        merged_model = reset_bn_stats(merged_model, combine_loader('mnist', 'cifar10'), device)
        print(coef, evaluate(model_list, merged_model, ['mnist', 'cifar10'], 32, 4, False))
        # print(evaluate_merged_model_robsust(merged_model, 'mnist', 32, 4, False, 0))
        # print(evaluate_merged_model_robsust(merged_model, 'cifar10', 32, 4, False, 1))


# *************slerp**************
print("slerp\n")
model_mnist = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/mnist/version_0/checkpoints/epoch=0-step=469-train_loss_adv=0.09.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/mnist/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_mnist, 'mnist', 32, 4, False))
# evaluate_upstream_model_robsust(model_mnist, 'mnist', 32, 4, False)
model_cifar10 = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/cifar10/version_0/checkpoints/epoch=103-step=40664-train_loss_adv=0.45.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/cifar10/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_cifar10, 'cifar10', 32, 4, False))
# evaluate_upstream_model_robsust(model_cifar10, 'cifar10', 32, 4, False)

pre_model = torch.load((f'/data4/xxx/0407/finetune/simclr/cifar10/r152_1x_sk0_pretrain.pth'))
model_list = [model_mnist.model, model_cifar10.model]
ckpt_list = [model_mnist.model.model.net.state_dict(), model_cifar10.model.model.net.state_dict()]

for degree in range(5, 6, 1):
    degree = degree / 10
    merge_ckpt = SLERP(ckpt_list[0], ckpt_list[1], degree)
    merged_model = simclr_merged_model(model_list, False)
    merged_model.load_weight(merge_ckpt)
    merged_model = reset_bn_stats(merged_model, combine_loader('mnist', 'cifar10'), device)
    print(degree, evaluate(model_list, merged_model, ['mnist', 'cifar10'], 32, 4, False))
    print(evaluate_merged_model_robsust(merged_model, 'mnist', 32, 4, False, 0))
    print(evaluate_merged_model_robsust(merged_model, 'cifar10', 32, 4, False, 1))

# *************adalayer************
model_mnist = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/mnist/version_0/checkpoints/epoch=0-step=469-train_loss_adv=0.09.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/mnist/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_mnist, 'mnist', 32, 4, False))
# evaluate_upstream_model_robsust(model_mnist, 'mnist', 32, 4, False)
model_cifar10 = LitPGDTrainer.load_from_checkpoint('/data4/xxx/0407/at_resnet/r152_1x/cifar10/version_0/checkpoints/epoch=103-step=40664-train_loss_adv=0.45.ckpt', 
                                           model = torch.load((f'/data4/xxx/0407/finetune/simclr/cifar10/r152_1x_sk0_pretrain.pth')))
# print(evaluate_upstream_model(model_cifar10, 'cifar10', 32, 4, False))
# evaluate_upstream_model_robsust(model_cifar10, 'cifar10', 32, 4, False)

# pre_model = torch.load((f'/data4/xxx/0407/finetune/simclr/cifar10/r152_1x_sk0_pretrain.pth'))
# model_list = [model_mnist.model, model_cifar10.model]
# ckpt_list = [TaskVector(pre_model.model.net.state_dict(), model_mnist.model.model.net.state_dict()), 
#              TaskVector(pre_model.model.net.state_dict(), model_cifar10.model.model.net.state_dict()), 
#              ]
# test_loader_list = [get_loader('mnist', 32, 1, 4, False, None)[2], get_loader('cifar10', 32, 1, 4, False, None)[2]]
# adamerging_task = AdaMerging_Layer('simclr', model_list, pre_model.model.net, ckpt_list, 1e-3, test_loader_list)
# name = 'mnist_cifar10'
# logger = TensorBoardLogger(f'/data4/xxx/0407/adamerging_layer/simclr/adv', name = name)
# import pytorch_lightning as pl
# trainer = pl.Trainer(max_steps= 300, devices=1, accelerator="gpu", deterministic=True, logger=logger)
# trainer.fit(adamerging_task)
# merged_ckpt = adamerging_task.get_merged_ckpt()
# merged_coff = adamerging_task.last_step_lambda
# merged_loss = adamerging_task.last_step_loss
# merged_model = simclr_merged_model(model_list, False)
# merged_model.load_weight(merged_ckpt)
# merged_model = reset_bn_stats(merged_model, combine_loader('mnist', 'cifar10'), device)
# print(evaluate(model_list, merged_model, ['mnist', 'cifar10'], 32, 4, False))
# print(evaluate_merged_model_robsust(merged_model, 'mnist', 32, 4, False, 0))
# print(evaluate_merged_model_robsust(merged_model, 'cifar10', 32, 4, False, 1))