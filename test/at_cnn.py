# domain: cv
# model: resnet50, cvit-b/16, clip-b/16
# algorithm: soups, ta, tadare, ties, tiesdare, slerp, fisher, regmean, adamerging
# type: single, multi
# dataset: gtsrb, sun397

from pytorch_lightning import Trainer
import torch
import sys

import torch
sys.path.append('..')
sys.path.append('../..')
from utils.files import write_to_csv
from merge.merge_class import simclr_merged_model
from merge.merge_utils import evaluate, evaluate_upstream_model
from algorithms.slerp import SLERP
from models.simclr.ft_class import ImageClassifier
from data.data_utils_1 import get_loader
from autoattack import AutoAttack
from pgd_adv_train_class import LitPGDTrainer
from pytorch_lightning.loggers import TensorBoardLogger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

type_list = ['r50_1x', 'r101_1x', 'r152_1x']
datasets_list = ['svhn', 'cifar10', 'mnist']
max_steps_list = [50000, 50000, 10000]

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training script with configurable parameters")
    
    # 添加必须的参数
    parser.add_argument(
        "--model_type", 
        type=str, 
        required=True,
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
    )
    parser.add_argument(
        "--steps", 
        type=int, 
    )
        
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Model: {args.model_type}, Dataset: {args.dataset}, Steps: {args.steps}")
    model_type = args.model_type
    dataset = args.dataset
    steps = args.steps

    model = torch.load((f'/data4/whj/0407/finetune/simclr/{dataset}/{model_type}_sk0_pretrain.pth'))
    train_loader, test_loader, val_loader = get_loader(dataset, 128, 1, 4, False)
    from torch.utils.data import ConcatDataset, DataLoader
    train_val_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    combined_loader = DataLoader(
        train_val_dataset,
        batch_size=128,        # 保持与原 loader 相同的 batch_size
        shuffle=True,          # 通常需要打乱
        num_workers=4,         # 与原 loader 相同的 workers
        pin_memory=True       # 与原 loader 相同的配置
    )
    lit_model = LitPGDTrainer(model=model, lr=1e-3, eps=8/255, alpha=2/255, pgd_steps=7)
    logger_path = f'/data4/whj/0407/at_resnet/{model_type}'
    logger = TensorBoardLogger(logger_path, name = dataset)
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
    model_checkpoint_train_loss = ModelCheckpoint(
        save_last=True,
        save_top_k=5,
        monitor="train_loss_adv",
        filename="{epoch:d}-{step:d}-{train_loss_adv:.2f}",
    )
    # # 用对抗样本继续训练
    trainer = Trainer(max_steps=steps, devices=1, logger=logger, callbacks=model_checkpoint_train_loss, precision="16-mixed")
    trainer.fit(lit_model, combined_loader)
    # import pdb; pdb.set_trace()
    # lit_model = LitPGDTrainer.load_from_checkpoint('/data4/whj/0407/adv_ft/r50_1x/cifar100/version_1/checkpoints/epoch=9-step=3130.ckpt', model = model)
    # evaluate_upstream_model_robsust(lit_model, dataset, 32, 4, False)


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
    log_path = 'cnn.csv'
    norm = 'Linf'
    adversary = AutoAttack(forward_pass, norm = norm, eps = 8. / 255., log_path = log_path, verbose = True)
    # ckeanacc: all test
    cleanacc = adversary.clean_accuracy(x_test, y_test, bs=250)
    print(cleanacc)
    # robustacc: all test
    n_ex = 1000
    with torch.no_grad(): 
        # individual
        with open(log_path, "a") as f:
            adversary = AutoAttack(forward_pass, norm = norm, eps = 8. / 255., log_path = log_path, verbose = True, version='standard')
            _ = adversary.run_standard_evaluation(
                x_test[:n_ex], y_test[:n_ex], bs=250, return_labels=True)
    return 




    