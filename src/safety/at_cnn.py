
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
from data.data_utils import get_loader
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

    model = torch.load((f'/data4/xxx/0407/finetune/simclr/{dataset}/{model_type}_sk0_pretrain.pth'))
    train_loader, test_loader, val_loader = get_loader(dataset, 128, 1, 4, False)
    from torch.utils.data import ConcatDataset, DataLoader
    train_val_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    combined_loader = DataLoader(
        train_val_dataset,
        batch_size=128,       
        shuffle=True,         
        num_workers=4,     
        pin_memory=True       
    )
    lit_model = LitPGDTrainer(model=model, lr=1e-3, eps=8/255, alpha=2/255, pgd_steps=7)
    logger_path = f'/data4/xxx/0407/at_resnet/{model_type}'
    logger = TensorBoardLogger(logger_path, name = dataset)
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
    model_checkpoint_train_loss = ModelCheckpoint(
        save_last=True,
        save_top_k=5,
        monitor="train_loss_adv",
        filename="{epoch:d}-{step:d}-{train_loss_adv:.2f}",
    )

    trainer = Trainer(max_steps=steps, devices=1, logger=logger, callbacks=model_checkpoint_train_loss, precision="16-mixed")
    trainer.fit(lit_model, combined_loader)