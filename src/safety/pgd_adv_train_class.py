import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import sys

import torch
sys.path.append('..')
sys.path.append('../..')
from merge.ft_class import CLIPImageClassifier
# from PGDAttack import PGDAttack
from torch.optim.lr_scheduler import StepLR

class LitPGDTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3, eps=8/255, alpha=2/255, pgd_steps=7):
        super().__init__()
        self.model = model
        if isinstance(model, CLIPImageClassifier):
        # 固定分类头参数（假设分类头参数名包含 'classifier'）
            self.model.model.visual_projection.weight.requires_grad = False
            self.model.classifier.weight.requires_grad = False
            print("分类头已固定！")
        self.lr = lr
        self.eps = eps
        self.alpha = alpha
        self.pgd_steps = pgd_steps

    def forward(self, x):
        return self.model(x)

    def pgd_attack(self, images, labels):
        images = images.detach().clone()
        ori_images = images.clone().detach()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        for _ in range(self.pgd_steps):
            images.requires_grad = True
            outputs = self(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            adv_images = images + self.alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach()
        for param in self.model.parameters():
            param.requires_grad_(True)
        self.model.train()
        return images

    def training_step(self, batch, batch_idx):
        images, labels = batch
        adv_images = self.pgd_attack(images, labels)
        outputs_adv = self(adv_images)
        outputs_clean = self(images)
        loss_adv = F.cross_entropy(outputs_adv, labels)
        loss_clean = F.cross_entropy(outputs_clean, labels)
        adv_ratio = loss_adv / (loss_adv + loss_clean + 1e-6)  # 对抗样本损失占比
        rand_weight = 0.5 + 0.5 * torch.rand(1).item()         # 随机成分 [0.5, 1.0)

        # 动态权重公式
        alpha = rand_weight * (1.0 + adv_ratio) 
        beta = 2.0 - alpha

        loss = alpha * loss_adv + beta * loss_clean
        # loss = loss_adv + loss_clean
        acc = (outputs_adv.argmax(dim=1) == labels).float().mean()
        self.log("train_loss_adv", loss_adv, prog_bar=True)
        self.log("train_loss_clean", loss_clean, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss_adv

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40000, 60000], gamma=0.1
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
