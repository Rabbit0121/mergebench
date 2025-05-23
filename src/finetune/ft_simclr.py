import argparse
import itertools
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import sys
sys.path.append('..')
from data.data_utils import *
from utils.files import *
from models.simclr.ft_class import *
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
pl.seed_everything(42, workers=True)

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_path', type=str, default='./configs/simclr.yaml', help='configs of finetuning')
args = parser.parse_args()
yaml_path = args.yaml_path

hyperparameters = get_hyperparameters(yaml_path)
# print(hyperparameters)
source = hyperparameters['source']

if source == 'simclr':
    keys = ['model_path', 'max_epochs', 'batch_size', 'learning_rate', 'devices',
             'percent', 'logger_path', 'result_path', 'dataset', 'num_workers', 'normalize', 'num_classes']
    values = [hyperparameters[key] for key in keys]
    model_path, max_epochs, batch_size, learning_rate, devices, percent, logger_path, result_path, dataset, num_workers, normalize, num_classes = values
 
    batch_sizes = []
    if (devices == 1):
        batch_sizes = [32, 64, 128]  
    else: batch_size = [8, 16, 32, 64, 128]

    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    param_combinations = list(itertools.product(batch_sizes, learning_rates))
    for item in param_combinations:
        batch_size = item[0]
        learning_rate = item[1]
        if(batch_size != 32 or learning_rate != 1e-2):
            continue
        print(batch_size, learning_rate)
        train_loader, test_loader, val_loader = get_loader(dataset, batch_size, percent, num_workers, normalize)
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch:02d}_{step}_{val_loss:.6f}',
            save_top_k=1,
            mode='min',
            save_last=True
        )
        

        model = ImageClassifier(model_path=model_path, lr = learning_rate, max_epochs = max_epochs, percent=percent, batch_size=batch_size, result_path=result_path, num_classes = num_classes, devices=devices)
        name = os.path.basename(model_path).split('.')[0]
        logger = TensorBoardLogger(logger_path, name = name)
        if not os.path.exists(f"{logger_path}"):
            os.makedirs(f"{logger_path}")
        torch.save(model, logger_path + f'/{name}_pretrain.pth')
        trainer = pl.Trainer(max_epochs=max_epochs, devices=devices, accelerator="gpu", deterministic=True, callbacks=[checkpoint_callback], logger=logger)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model=model, dataloaders=test_loader)