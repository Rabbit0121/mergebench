import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import sys
from merge_class import clip_merged_model
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
from ft_class import CLIPImageClassifier
from merge_clip import clipb16_model_path_list, clipb32_model_path_list, clipl14_model_path_list
from transformers import CLIPProcessor
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [clipb16_model_path_list, clipb32_model_path_list, clipl14_model_path_list]
type_list = ['clipb16', 'clipb32','clipl14']
processor_list = ['/data4/xxx/0407/models/hf_clip/clip-vit-base-patch16', 
                   '/data4/xxx/0407/models/hf_clip/clip-vit-base-patch32',
                   '/data4/xxx/0407/models/hf_clip/clip-vit-large-patch14']
pre_model_path_list = ['/data4/xxx/0407/ftnew/hf_clip/cifar10/clip-vit-base-patch16_pretrain.pth',
                  '/data4/xxx/0407/ftnew/hf_clip/cifar10/clip-vit-base-patch32_pretrain.pth',
                  '/data4/xxx/0407/ftnew/hf_clip/cifar10/clip-vit-large-patch14_pretrain.pth']
datasets_list  = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']

for model_path_list, model_type, model_processor_path, pre_model_path in zip(path_list, type_list, processor_list, pre_model_path_list):
    if(model_type != 'clipb16'):
        continue
    model_path_list = [model_path_list[2], model_path_list[3]]
    datasets_list = [datasets_list[2], datasets_list[3]]
    processor = CLIPProcessor.from_pretrained(model_processor_path + '/processor')
    test_loaders_list = [get_loader(dataset, 32, 1, 4, False, processor)[2] for dataset in datasets_list]
    acc_upstream_dict = {}
    
    
    for i in range(len(model_path_list)):
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model(CLIPImageClassifier.load_from_checkpoint(model_path_list[i]), datasets_list[i], 32, 4, False, processor)
    # 两两分配
    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            # adamerging-task
            for max_steps in [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]:
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
                model1 = CLIPImageClassifier.load_from_checkpoint(model_path_list[i])
                model2 = CLIPImageClassifier.load_from_checkpoint(model_path_list[j])
                model_list.append(model1)
                model_list.append(model2)
                ckpt_list.append(TaskVector(pre_model.model.vision_model.state_dict(), model1.model.vision_model.state_dict()))
                ckpt_list.append(TaskVector(pre_model.model.vision_model.state_dict(), model2.model.vision_model.state_dict()))
                # plus version
                # ties_ckpt_list = get_ties_ckpt(ckpt_list, pre_model.model.vision_model.state_dict())
                # adamerging_task = AdaMerging_Task('hf_clip', model_list, pre_model.model.vision_model, ties_ckpt_list, 1e-3, test_loader_list)
                adamerging_task = AdaMerging_Task('hf_clip', model_list, pre_model.model.vision_model, ckpt_list, 1e-3, test_loader_list)
                name = f'{dataset_list[0]}_{dataset_list[1]}'
                logger = TensorBoardLogger(f'/data4/xxx/0407/adamerging_task/hf_clip_discuss/{model_type}', name = name)
                trainer = pl.Trainer(max_steps= max_steps, devices=1, accelerator="gpu", deterministic=True, logger=logger)
                trainer.fit(adamerging_task)
                merged_ckpt = adamerging_task.get_merged_ckpt()
                merged_coff = adamerging_task.last_step_lambda
                merged_loss = adamerging_task.last_step_loss 
                merged_model = clip_merged_model(model_list, False)
                merged_model.load_weight(merged_ckpt)
                # evaluate
                acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
                _, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False, processor=processor)
                print(acc_upstream, acc_merged)

                headers=["datasets", "type", "max_steps", "merged_coff", "merged_loss", "acc_upstream", "acc_merged"]
                data={"datasets":model_type + f'_{dataset_list[0]}_{dataset_list[1]}', "acc_upstream":acc_upstream, "acc_merged":acc_merged, 
                        "type":"adamerging-task++", "max_steps":max_steps, "merged_coff":merged_coff, "merged_loss":merged_loss}
                result_path = '../results/new_merge_2/clip/adamerging_task_discuss.csv'
                write_to_csv(result_path, headers, data)
                