import sys
from merge_class import vit_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from utils.files import write_to_csv
from merge_utils import evaluate, evaluate_upstream_model
from algorithms.slerp import SLERP
from ft_class import ViTImageClassifier
from transformers import ViTImageProcessor
from merge_vit import modelb16_path_list, modelb32_path_list, modell32_path_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


path_list = [modelb16_path_list, modelb32_path_list, modell32_path_list]
type_list = ['vitb16', 'vitb32','vitl32']
processor_list  = ['/data4/xxx/0407/models/hf_vit/vit-base-patch16-224-in21k', '/data4/xxx/0407/models/hf_vit/vit-base-patch32-224-in21k', '/data4/xxx/0407/models/hf_vit/vit-large-patch32-224-in21k']
datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']

for model_path_list, model_type, model_processor_path in zip(path_list, type_list, processor_list):
    processor = ViTImageProcessor.from_pretrained(model_processor_path + '/processor')
    acc_upstream_dict = {}
    for i in range(len(model_path_list)):
        acc_upstream_dict[str(datasets_list[i])] = evaluate_upstream_model(ViTImageClassifier.load_from_checkpoint(model_path_list[i]), datasets_list[i], 32, 4, False, processor)


    for i in range(len(model_path_list)):
        for j in range(i+1, len(model_path_list), 1):
            model_list = []  
            ckpt_list = []
            dataset_list = []
            dataset_list.append(datasets_list[i])
            dataset_list.append(datasets_list[j])
            model1 = ViTImageClassifier.load_from_checkpoint(model_path_list[i])
            model2 = ViTImageClassifier.load_from_checkpoint(model_path_list[j])
            model_list.append(model1)
            model_list.append(model2)
            ckpt_list.append(model1.model.vit.state_dict())
            ckpt_list.append(model2.model.vit.state_dict())
            # slerp
            for degree in range(1, 10, 1):
                degree = degree / 10
                merge_ckpt = SLERP(ckpt_list[0], ckpt_list[1], degree)
                merged_model = vit_merged_model(model_list, False)
                merged_model.load_weight(merge_ckpt)
                # evaluate
                acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
                _, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False, processor=processor)
                print(acc_upstream, acc_merged)
                headers=["datasets", "degree", "acc_upstream", "acc_merged"]
                data={"datasets":model_type + str(dataset_list), "acc_upstream":acc_upstream, "acc_merged":acc_merged, "degree":degree}
                result_path = '../results/new_merge_2/vit/slerp.csv'
                write_to_csv(result_path, headers, data)