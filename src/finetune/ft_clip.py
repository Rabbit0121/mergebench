import itertools
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from transformers import CLIPProcessor, CLIPModel 
import sys


sys.path.append('..')
sys.path.append('../..')
from finetune.finetune_utils import get_fc_weight
from data.data_utils import get_loader
from utils.files import *
from ft_class import CLIPImageClassifier
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl
pl.seed_everything(42, workers=True)


dataset_list = [ 'gtsrb', 'cifar10', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist', 'svhn']
numclass_list = [43,      10,         397,      47,    100,        196,            10,       10,       10]
epoch_list = [   10,      10,         30,       70,    10,         35,             10,       10,       10]
batchsize_list = [128, 64]
lr_list = [1e-5, 2e-6]




param_grid = list(itertools.product(batchsize_list, lr_list))
for params in param_grid:
    batchsize = params[0]
    lr = params[1]
    model_processor_path = '/data4/xxx/0407/models/hf_clip/clip-vit-base-patch32'
    processor = CLIPProcessor.from_pretrained(model_processor_path + '/processor')
    for dataset, num_classes, epoch in zip(dataset_list, numclass_list, epoch_list):
        # print("参数组合 (batch_size, lr):")
        logger_path = f'/data4/xxx/0407/ftnew/hf_clip/{dataset}'
        result_path = f'/home/xxx/miniconda3/envs/0407/SecureMerge/src/test/fusionbench/{dataset}_results.csv'
        train_loader, test_loader, _ = get_loader(dataset, batchsize, 1, 4, 'False', processor)
        model = CLIPImageClassifier(model_path=model_processor_path, lr = lr, max_epochs = epoch, percent=1.0, batch_size=batchsize, result_path=result_path, num_classes = num_classes, devices=1)
        # clipmodel = CLIPModel.from_pretrained(model_processor_path + '/mdoel')
        fc_weight = get_fc_weight(dataset, processor, CLIPModel.from_pretrained(model_processor_path + '/model'))
        model.classifier.load_state_dict({'weight': fc_weight})
        
        name = os.path.basename(model_processor_path)
        logger = TensorBoardLogger(logger_path, name = name)
        if not os.path.exists(f"{logger_path}"):
            os.makedirs(f"{logger_path}")
        torch.save(model, logger_path + f'/{name}_pretrain.pth')
        trainer = pl.Trainer(max_epochs = epoch, devices=1, accelerator="gpu", deterministic=True, logger=logger)
        trainer.fit(model, train_loader)
        trainer.test(model=model, dataloaders=test_loader)


    # model_processor_path = '/data4/xxx/0407/models/hf_clip/clip-vit-base-patch16'
    # processor = CLIPProcessor.from_pretrained(model_processor_path + '/processor')
    # for dataset,num_classes, epoch in zip(dataset_list, numclass_list, epoch_list):
    #     logger_path = f'/data4/xxx/0407/ftnew/hf_clip/{dataset}'
    #     result_path = f'/home/xxx/miniconda3/envs/0407/SecureMerge/src/test/fusionbench/{dataset}_results.csv'
    #     train_loader, test_loader, _ = get_loader(dataset, batchsize, 1, 4, 'False', processor)
    #     model = CLIPImageClassifier(model_path=model_processor_path, lr = lr, max_epochs = epoch, percent=1.0, batch_size=batchsize, result_path=result_path, num_classes = num_classes, devices=1)
    #     # clipmodel = CLIPModel.from_pretrained(model_processor_path + '/mdoel')
    #     fc_weight = get_fc_weight(dataset, processor, CLIPModel.from_pretrained(model_processor_path + '/model'))
    #     model.classifier.load_state_dict({'weight': fc_weight})
        
    #     name = os.path.basename(model_processor_path)
    #     logger = TensorBoardLogger(logger_path, name = name)
    #     if not os.path.exists(f"{logger_path}"):
    #         os.makedirs(f"{logger_path}")
    #     torch.save(model, logger_path + f'/{name}_pretrain.pth')
    #     trainer = pl.Trainer(max_epochs = epoch, devices=1, accelerator="gpu", deterministic=True, logger=logger)
    #     trainer.fit(model, train_loader)
    #     trainer.test(model=model, dataloaders=test_loader)


    # model_processor_path = '/data4/xxx/0407/models/hf_clip/clip-vit-base-large14'
    # processor = CLIPProcessor.from_pretrained(model_processor_path + '/processor')
    # for dataset,num_classes, epoch in zip(dataset_list, numclass_list, epoch_list):
    #     logger_path = f'/data4/xxx/0407/ftnew/hf_clip/{dataset}'
    #     result_path = f'/home/xxx/miniconda3/envs/0407/SecureMerge/src/test/fusionbench/{dataset}_results.csv'
    #     train_loader, test_loader, _ = get_loader(dataset, batchsize, 1, 4, 'False', processor)
    #     model = CLIPImageClassifier(model_path=model_processor_path, lr = lr, max_epochs = epoch, percent=1.0, batch_size=batchsize, result_path=result_path, num_classes = num_classes, devices=1)
    #     # clipmodel = CLIPModel.from_pretrained(model_processor_path + '/mdoel')
    #     fc_weight = get_fc_weight(dataset, processor, CLIPModel.from_pretrained(model_processor_path + '/model'))
    #     model.classifier.load_state_dict({'weight': fc_weight})
        
    #     name = os.path.basename(model_processor_path)
    #     logger = TensorBoardLogger(logger_path, name = name)
    #     if not os.path.exists(f"{logger_path}"):
    #         os.makedirs(f"{logger_path}")
    #     torch.save(model, logger_path + f'/{name}_pretrain.pth')
    #     trainer = pl.Trainer(max_epochs = epoch, devices=1, accelerator="gpu", deterministic=True, logger=logger)
    #     trainer.fit(model, train_loader)
    #     trainer.test(model=model, dataloaders=test_loader)