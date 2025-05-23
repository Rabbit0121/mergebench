import sys
from merge_class import clip_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from utils.files import write_to_csv
from merge_utils import evaluate, evaluate_upstream_model
from algorithms.task_arithmetic import TaskVector
from algorithms.dare import DARE
from ft_class import CLIPImageClassifier
from transformers import CLIPProcessor
from merge_clip import clipb16_model_path_list, clipb32_model_path_list, clipl14_model_path_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_list = [clipb16_model_path_list, clipb32_model_path_list, clipl14_model_path_list]
type_list = ['clipb16', 'clipb32','clipl14']
processor_list  = ['/data4/xxx/0407/models/hf_clip/clip-vit-base-patch16', 
                   '/data4/xxx/0407/models/hf_clip/clip-vit-base-patch32',
                   '/data4/xxx/0407/models/hf_clip/clip-vit-large-patch14']
pre_model_list = [torch.load('/data4/xxx/0407/ftnew/hf_clip/cifar10/clip-vit-base-patch16_pretrain.pth'),
                  torch.load('/data4/xxx/0407/ftnew/hf_clip/cifar10/clip-vit-base-patch32_pretrain.pth'),
                  torch.load('/data4/xxx/0407/ftnew/hf_clip/cifar10/clip-vit-large-patch14_pretrain.pth')]

datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']

for model_path_list, model_type, model_processor_path, pre_model in zip(path_list, type_list, processor_list, pre_model_list):
    processor = CLIPProcessor.from_pretrained(model_processor_path + '/processor')
    acc_upstream_dict = {}
    for i in range(len(model_path_list)):
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model(CLIPImageClassifier.load_from_checkpoint(model_path_list[i]), datasets_list[i], 32, 4, False, processor)
        # print(acc_upstream_dict)

    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            model_list = []  
            ckpt_list = []
            dataset_list = []
            dataset_list.append(datasets_list[i])
            dataset_list.append(datasets_list[j])
            model1 = CLIPImageClassifier.load_from_checkpoint(model_path_list[i])
            model2 = CLIPImageClassifier.load_from_checkpoint(model_path_list[j])
            model_list.append(model1)
            model_list.append(model2)
            for dare_rate in range(1, 10, 1):
                if(dare_rate != 1):
                    continue
                dare_rate = dare_rate / 10
                ckpt_list = []
                ckpt_list.append(DARE(TaskVector(pre_model.model.vision_model.state_dict(), model1.model.vision_model.state_dict()), mask_rate = dare_rate, use_rescale = True, mask_strategy = 'magnitude'))
                ckpt_list.append(DARE(TaskVector(pre_model.model.vision_model.state_dict(), model2.model.vision_model.state_dict()), mask_rate = dare_rate, use_rescale = True, mask_strategy = 'magnitude'))
                sum_taskvector = sum(ckpt_list)
                for coef in range(1, 10, 1):
                    coef = coef /10
                    merge_ckpt = sum_taskvector.apply_to(pre_model.model.vision_model.state_dict(), scaling_coef = coef)
                    merged_model = clip_merged_model(model_list, False)
                    merged_model.load_weight(merge_ckpt)
                    # evaluate
                    acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
                    _, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False, processor=processor)
                    print(acc_upstream, acc_merged)
                    headers=["datasets","dare_rate", "lambda", "acc_upstream", "acc_merged"]
                    data={"datasets":model_type + str(dataset_list), "acc_upstream":acc_upstream, "acc_merged":acc_merged, "lambda": coef, "dare_rate":dare_rate}
                    result_path = '../results/new_merge_2/clip/tadare.csv'
                    write_to_csv(result_path, headers, data)