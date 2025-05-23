import itertools
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from transformers import BertTokenizer
import sys


sys.path.append('..')
sys.path.append('../..')
from finetune.finetune_utils import get_fc_weight
from data.data_utils import get_loader_nlp
from utils.files import *
from ft_class import BertSequenceClassifier
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl
pl.seed_everything(42, workers=True)

# metric_list = [acc,  mas'core,(mis)matacc,   f1/acc,  acc,   f1/acc, acc,   acc]
datasets_list = ['sst2', 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'wnli']
numclass_list = [2,       2,      3,      2,      2,      2,     2,     2    ]
percent_list =  [1,       1,      1,      1,      1,      0.5,   1,     1]
epoch_list =    [3,       3,      3,      3,      3,      3,     5,     3]
batchsize_list = [32, 16]
lr_list = [1e-5, 1e-6]

model_tokenizer_path = '/data4/xxx/0407/models/hf_bert/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_tokenizer_path + '/tokenizer')
param_grid = list(itertools.product(batchsize_list, lr_list))
for dataset, num_classes, epoch, percent in zip(datasets_list, numclass_list, epoch_list, percent_list):
    logger_path = f'/data4/xxx/0407/ftnew/hf_bert/{dataset}'
    result_path = f'/home/xxx/miniconda3/envs/0407/SecureMerge/src/test/fusionbench/{dataset}_results.csv'
    if(dataset in ['sst2', 'cola', 'mnli', 'qnli', 'qqp', 'wnli', 'mrpc']):
        continue
    if(dataset == 'mrpc'):
        param_grid = list(itertools.product([32, 16], [5e-5, 5e-6]))
    if(dataset == 'rte'):
        param_grid = list(itertools.product([16], [5e-5, 1e-4]))
    for params in param_grid:
        batchsize = params[0]
        lr = params[1]
        print(dataset, batchsize, lr)
        # import pdb; pdb.set_trace()
        train_loader, test_loader, _ = get_loader_nlp(dataset_name=dataset, batch_size=batchsize, percent=percent, num_workers=4, tokenizer=tokenizer)
        model = BertSequenceClassifier(model_tokenizer_path, lr=lr, max_epochs=epoch, percent=percent, batch_size=batchsize, num_classes=num_classes, result_path=result_path, devices=1)
        name = os.path.basename(model_tokenizer_path)
        logger = TensorBoardLogger(logger_path, name = name)
        if not os.path.exists(f"{logger_path}"):
            os.makedirs(f"{logger_path}")
        torch.save(model, logger_path + f'/{name}_pretrain.pth')
        trainer = pl.Trainer(max_epochs = epoch, devices=1, accelerator="gpu", deterministic=True, logger=logger, precision="16-mixed")
        trainer.fit(model, train_loader)
        trainer.test(model=model, dataloaders=test_loader)
        # import pdb; pdb.set_trace()


# model_tokenizer_path = '/data4/xxx/0407/models/hf_bert/bert-large-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_tokenizer_path + '/tokenizer')
# param_grid = list(itertools.product(batchsize_list, lr_list))
# for dataset, num_classes, epoch, percent in zip(datasets_list, numclass_list, epoch_list, percent_list):
#     logger_path = f'/data4/xxx/0407/ftnew/hf_bert/{dataset}'
#     result_path = f'/home/xxx/miniconda3/envs/0407/SecureMerge/src/test/fusionbench/{dataset}_results.csv'
#     for params in param_grid:
#         batchsize = params[0]
#         lr = params[1]
#         print(dataset, batchsize, lr)
#         # import pdb; pdb.set_trace()
#         train_loader, test_loader, _ = get_loader_nlp(dataset_name=dataset, batch_size=batchsize, percent=percent, num_workers=4, tokenizer=tokenizer)
#         model = BertSequenceClassifier(model_tokenizer_path, lr=lr, max_epochs=epoch, percent=percent, batch_size=batchsize, num_classes=num_classes, result_path=result_path, devices=1)
#         name = os.path.basename(model_tokenizer_path)
#         logger = TensorBoardLogger(logger_path, name = name)
#         if not os.path.exists(f"{logger_path}"):
#             os.makedirs(f"{logger_path}")
#         torch.save(model, logger_path + f'/{name}_pretrain.pth')
#         trainer = pl.Trainer(max_epochs = epoch, devices=1, accelerator="gpu", deterministic=True, logger=logger, precision="16-mixed")
#         trainer.fit(model, train_loader)
#         trainer.test(model=model, dataloaders=test_loader)
#         import pdb; pdb.set_trace()