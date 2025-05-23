import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import sys
from merge_class import vit_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from src.algorithms.ties import get_ties_ckpt
from src.data.data_utils import get_loader
from utils.files import write_to_csv
from merge_utils_1 import evaluate, evaluate_upstream_model
from algorithms.adamerging import AdaMerging_Task
from algorithms.task_arithmetic import TaskVector
from algorithms.task_arithmetic import TaskVector
from ft_class import ViTImageClassifier
from merge_vit import modelb16_path_list, modelb32_path_list, modell32_path_list
from transformers import ViTImageProcessor



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [modelb16_path_list, modelb32_path_list, modell32_path_list]
type_list = ['vitb16', 'vitb32','vitl32']
processor_list  = ['/data4/xxx/0407/models/hf_vit/vit-base-patch16-224-in21k', 
                   '/data4/xxx/0407/models/hf_vit/vit-base-patch32-224-in21k', 
                   '/data4/xxx/0407/models/hf_vit/vit-large-patch32-224-in21k']
pre_model_path_list = ['/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-base-patch16-224-in21k_pretrain.pth',
                  '/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-base-patch32-224-in21k_pretrain.pth',
                  '/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-large-patch32-224-in21k_pretrain.pth']
datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']

for model_path_list, model_type, model_processor_path, pre_model_path in zip(path_list, type_list, processor_list, pre_model_path_list):
    processor = ViTImageProcessor.from_pretrained(model_processor_path + '/processor')
    test_loaders_list = [get_loader(dataset, 32, 1, 4, False, processor)[2] for dataset in datasets_list]
    acc_upstream_dict = {}
    model_path_list = model_path_list[:2]
    for i in range(len(model_path_list)):
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model(ViTImageClassifier.load_from_checkpoint(model_path_list[i]), datasets_list[i], 32, 4, False, processor)
    # 两两分配
    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            # adamerging-task
            for max_steps in [100, 300, 500, 700, 900, 1000]:
                if(max_steps != 900):
                    continue
                pre_model = torch.load(pre_model_path)
                model_list = []  
                ckpt_list = []
                dataset_list = []
                test_loader_list = []
                dataset_list.append(datasets_list[i])
                dataset_list.append(datasets_list[j])
                test_loader_list.append(test_loaders_list[i])
                test_loader_list.append(test_loaders_list[j])
                model1 = ViTImageClassifier.load_from_checkpoint(model_path_list[i])
                model2 = ViTImageClassifier.load_from_checkpoint(model_path_list[j])
                model_list.append(model1)
                model_list.append(model2)
                ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model1.model.vit.state_dict()))
                ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model2.model.vit.state_dict()))
                # plus version
                ties_ckpt_list = get_ties_ckpt(ckpt_list, pre_model.model.vit.state_dict())
                adamerging_task = AdaMerging_Task('hf_vit', model_list, pre_model.model.vit, ties_ckpt_list, 1e-3, test_loader_list)
                # adamerging_task = AdaMerging_Task('hf_vit', model_list, pre_model.model.vit, ckpt_list, 1e-3, test_loader_list)
                name = f'{dataset_list[0]}_{dataset_list[1]}'
                logger = TensorBoardLogger(f'/data4/xxx/0407/adamerging_task++/hf_vit/{model_type}', name = name)
                trainer = pl.Trainer(max_steps= max_steps, devices=1, accelerator="gpu", deterministic=True, logger=logger)
                trainer.fit(adamerging_task)
                merged_ckpt = adamerging_task.get_merged_ckpt()
                merged_coff = adamerging_task.last_step_lambda
                merged_loss = adamerging_task.last_step_loss 
                merged_model = vit_merged_model(model_list, False)
                merged_model.load_weight(merged_ckpt)
                # evaluate
                acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
                _, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False, processor=processor)
                print(acc_upstream, acc_merged)

                headers=["datasets", "type", "max_steps", "merged_coff", "merged_loss", "acc_upstream", "acc_merged"]
                data={"datasets":model_type + f'_{dataset_list[0]}_{dataset_list[1]}', "acc_upstream":acc_upstream, "acc_merged":acc_merged, 
                        "type":"adamerging-task++", "max_steps":max_steps, "merged_coff":merged_coff, "merged_loss":merged_loss}
                result_path = '../results/new_merge_2/vit/adamerging_task++.csv'
                write_to_csv(result_path, headers, data)
                # import pdb; pdb.set_trace()
                