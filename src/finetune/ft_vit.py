import itertools
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from transformers import ViTImageProcessor, CLIPModel 
import sys
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append('..')
sys.path.append('../..')
from finetune.finetune_utils import get_fc_weight
from data.data_utils_1 import get_loader
from utils.files import *
from ft_class import ViTImageClassifier
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl
pl.seed_everything(42, workers=True)


dataset_list = [ 'gtsrb', 'cifar10', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist', 'svhn']
numclass_list = [43,      10,         397,      47,    100,        196,            10,       10,       10]
epoch_list = [   10,      10,         30,       70,    10,         35,             10,       10,       10]
batchsize_list = [32, 16]
lr_list = [1e-5, 2e-6]



model_processor_path = '/data4/xxx/0407/models/hf_vit/vit-base-patch32-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_processor_path + '/processor')
for dataset,num_classes, epoch in zip(dataset_list, numclass_list, epoch_list):
    if (dataset not in batchsize_lr.keys()):
        continue
    logger_path = f'/data4/xxx/0407/ftnew/hf_vit/{dataset}'
    result_path = f'/home/xxx/miniconda3/envs/0407/SecureMerge/src/test/fusionbench/{dataset}_results.csv'
    for item in batchsize_lr[dataset]:
        batchsize = item[0]
        lr = item[1]
        print(dataset, batchsize, lr)
        if(batchsize != 64):
            continue
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch:02d}_{step}_{val_loss:.6f}',
            save_top_k=1,
            mode='min',
            save_last=True
        )
        # import pdb; pdb.set_trace()
        train_loader, test_loader, val_loader = get_loader(dataset, batchsize, 1, 4, 'False', processor)
        model = ViTImageClassifier(model_path=model_processor_path, lr = lr, max_epochs = epoch, percent=1.0, batch_size=batchsize, result_path=result_path, num_classes = num_classes, devices=1)
        name = os.path.basename(model_processor_path)
        logger = TensorBoardLogger(logger_path, name = name)
        if not os.path.exists(f"{logger_path}"):
            os.makedirs(f"{logger_path}")
        torch.save(model, logger_path + f'/{name}_pretrain.pth')
        trainer = pl.Trainer(max_epochs = epoch, devices=1, accelerator="gpu", deterministic=True, callbacks=[checkpoint_callback], logger=logger)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model=model, dataloaders=test_loader)


# model_processor_path = '/data4/xxx/0407/models/hf_vit/vit-base-patch16-224-in21k'
# processor = ViTImageProcessor.from_pretrained(model_processor_path + '/processor')
# for dataset,num_classes, epoch in zip(dataset_list, numclass_list, epoch_list):
#     if (dataset not in batchsize_lr.keys()):
#         continue

#     logger_path = f'/data4/xxx/0407/ftnew/hf_vit/{dataset}'
#     result_path = f'/home/xxx/miniconda3/envs/0407/SecureMerge/src/test/fusionbench/{dataset}_results.csv'
#     for item in batchsize_lr[dataset]:
#         batchsize = item[0]
#         lr = item[1]
#         print(dataset, batchsize, lr)
#         # import pdb; pdb.set_trace()
#         train_loader, test_loader, _ = get_loader(dataset, batchsize, 1, 4, 'False', processor)
#         model = ViTImageClassifier(model_path=model_processor_path, lr = lr, max_epochs = epoch, percent=1.0, batch_size=batchsize, result_path=result_path, num_classes = num_classes, devices=1)
#         name = os.path.basename(model_processor_path)
#         logger = TensorBoardLogger(logger_path, name = name)
#         if not os.path.exists(f"{logger_path}"):
#             os.makedirs(f"{logger_path}")
#         torch.save(model, logger_path + f'/{name}_pretrain.pth')
#         trainer = pl.Trainer(max_epochs = epoch, devices=1, accelerator="gpu", deterministic=True, logger=logger)
#         trainer.fit(model, train_loader)
#         trainer.test(model=model, dataloaders=test_loader)



# model_processor_path = '/data4/xxx/0407/models/hf_vit/vit-large-patch32-224-in21k'
# processor = ViTImageProcessor.from_pretrained(model_processor_path + '/processor')
# for dataset,num_classes, epoch in zip(dataset_list, numclass_list, epoch_list):
#     if (dataset not in batchsize_lr.keys()):
#         continue

#     logger_path = f'/data4/xxx/0407/ftnew/hf_vit/{dataset}'
#     result_path = f'/home/xxx/miniconda3/envs/0407/SecureMerge/src/test/fusionbench/{dataset}_results.csv'
#     for item in batchsize_lr[dataset]:
#         batchsize = item[0]
#         lr = item[1]
#         print(dataset, batchsize, lr)
#         # import pdb; pdb.set_trace()
#         train_loader, test_loader, _ = get_loader(dataset, batchsize, 1, 4, 'False', processor)
#         model = ViTImageClassifier(model_path=model_processor_path, lr = lr, max_epochs = epoch, percent=1.0, batch_size=batchsize, result_path=result_path, num_classes = num_classes, devices=1)
#         name = os.path.basename(model_processor_path)
#         logger = TensorBoardLogger(logger_path, name = name)
#         if not os.path.exists(f"{logger_path}"):
#             os.makedirs(f"{logger_path}")
#         torch.save(model, logger_path + f'/{name}_pretrain.pth')
#         trainer = pl.Trainer(max_epochs = epoch, devices=1, accelerator="gpu", deterministic=True, logger=logger)
#         trainer.fit(model, train_loader)
#         trainer.test(model=model, dataloaders=test_loader)
