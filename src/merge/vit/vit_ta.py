import sys
from merge_class import vit_merged_model
import torch
sys.path.append('..')
sys.path.append('../..')
from utils.files import write_to_csv
from merge_utils import evaluate, evaluate_upstream_model
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
pre_model_list = [torch.load('/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-base-patch16-224-in21k_pretrain.pth'),
                  torch.load('/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-base-patch32-224-in21k_pretrain.pth'),
                  torch.load('/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-large-patch32-224-in21k_pretrain.pth')]

datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']
for model_path_list, model_type, model_processor_path, pre_model in zip(path_list, type_list, processor_list, pre_model_list):
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

        ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model1.model.vit.state_dict()))
        ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model2.model.vit.state_dict()))
        # taskvector
        sum_taskvector = sum(ckpt_list)
        for coef in range(1, 10, 1):
            coef = coef /10
            merge_ckpt = sum_taskvector.apply_to(pre_model.model.vit.state_dict(), scaling_coef = coef)
            merged_model = vit_merged_model(model_list, False)
            merged_model.load_weight(merge_ckpt)
            # evaluate
            acc_upstream = [acc_upstream_dict[dataset_list[0]], acc_upstream_dict[dataset_list[1]]]
            _, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False, processor=processor)
            print(acc_upstream, acc_merged)
            headers=["datasets", "lambda", "acc_upstream", "acc_merged"]
            data={"datasets":model_type + str(dataset_list), "acc_upstream":acc_upstream, "acc_merged":acc_merged, "lambda": coef}
            result_path = '../results/new_merge_2/vit/ta.csv'
            write_to_csv(result_path, headers, data)












# model_path_list = [
#                     '/data4/xxx/0407/ftnew/hf_vit/gtsrb/vit-base-patch16-224-in21k/version_1/checkpoints/epoch=9-step=13320.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-base-patch16-224-in21k/version_0/checkpoints/epoch=9-step=12500.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/svhn/vit-base-patch16-224-in21k/version_0/checkpoints/epoch=9-step=18320.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/sun397/vit-base-patch16-224-in21k/version_0/checkpoints/epoch=29-step=61200.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/dtd/vit-base-patch16-224-in21k/version_2/checkpoints/epoch=69-step=4130.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/cifar100/vit-base-patch16-224-in21k/version_2/checkpoints/epoch=9-step=12500.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/stanfordcars/vit-base-patch16-224-in21k/version_0/checkpoints/last.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/eurosat/vit-base-patch16-224-in21k/version_0/checkpoints/epoch=9-step=5070.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/mnist/vit-base-patch16-224-in21k/version_3/checkpoints/epoch=9-step=30000.ckpt'
#                   ] 
# datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']
# model_type = 'vitb16'
# model_processor_path = '/data4/xxx/0407/models/hf_vit/vit-base-patch16-224-in21k'
# pre_model = torch.load('/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-base-patch16-224-in21k_pretrain.pth')
# processor = ViTImageProcessor.from_pretrained(model_processor_path + '/processor')
# # 两两分配
# for i in range(len(model_path_list)):
#     # model1 = ViTImageClassifier.load_from_checkpoint(model_path_list[i])
#     # print('ok')
#     # continue
#     for j in range(i+1, len(model_path_list), 1):
#         model_list = []  
#         ckpt_list = []
#         dataset_list = []
#         dataset_list.append(datasets_list[i])
#         dataset_list.append(datasets_list[j])
#         model1 = ViTImageClassifier.load_from_checkpoint(model_path_list[i])
#         model2 = ViTImageClassifier.load_from_checkpoint(model_path_list[j])
#         model_list.append(model1)
#         model_list.append(model2)
#         # print(model1.state_dict().keys())
#         # print(model1.model.vit.state_dict().keys())
#         # print(model1.model.classifier.state_dict().keys())
#         # import pdb; pdb.set_trace()
#         ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model1.model.vit.state_dict()))
#         ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model2.model.vit.state_dict()))
#         # taskvector
#         sum_taskvector = sum(ckpt_list)
#         for coef in range(1, 10, 1):
#             coef = coef /10
#             merge_ckpt = sum_taskvector.apply_to(pre_model.model.vit.state_dict(), scaling_coef = coef)
#             merged_model = vit_merged_model(model_list, False)
#             merged_model.load_weight(merge_ckpt)
#             # evaluate
#             acc_upstream, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False, processor=processor)
#             print(acc_upstream, acc_merged)
#             headers=["datasets", "lambda", "acc_upstream", "acc_merged"]
#             data={"datasets":model_type + str(dataset_list), "acc_upstream":acc_upstream, "acc_merged":acc_merged, "lambda": coef}
#             result_path = '../results/new_merge_2/vit/ta.csv'
#             write_to_csv(result_path, headers, data)


# model_path_list = [
#                     '/data4/xxx/0407/ftnew/hf_vit/gtsrb/vit-base-patch32-224-in21k/version_1/checkpoints/epoch=9-step=13320.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-base-patch32-224-in21k/version_0/checkpoints/epoch=9-step=12500.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/svhn/vit-base-patch32-224-in21k/version_3/checkpoints/epoch=9-step=18320.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/sun397/vit-base-patch32-224-in21k/version_2/checkpoints/epoch=29-step=122370.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/dtd/vit-base-patch32-224-in21k/version_1/checkpoints/epoch=69-step=8260.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/cifar100/vit-base-patch32-224-in21k/version_2/checkpoints/epoch=9-step=12500.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/stanfordcars/vit-base-patch32-224-in21k/version_0/checkpoints/epoch=41_step=17136_val_loss=1.354956.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/eurosat/vit-base-patch32-224-in21k/version_3/checkpoints/epoch=9-step=10130.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/mnist/vit-base-patch32-224-in21k/version_2/checkpoints/epoch=9-step=30000.ckpt'
#                   ] 
# datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']
# model_type = 'vitb32'
# model_processor_path = '/data4/xxx/0407/models/hf_vit/vit-base-patch32-224-in21k'
# pre_model = torch.load('/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-base-patch32-224-in21k_pretrain.pth')
# processor = ViTImageProcessor.from_pretrained(model_processor_path + '/processor')
# # 两两分配
# for i in range(len(model_path_list)):
#     # model1 = ViTImageClassifier.load_from_checkpoint(model_path_list[i])
#     # print('ok')
#     # continue
#     for j in range(i+1, len(model_path_list), 1):
#         model_list = []  
#         ckpt_list = []
#         dataset_list = []
#         dataset_list.append(datasets_list[i])
#         dataset_list.append(datasets_list[j])
#         model1 = ViTImageClassifier.load_from_checkpoint(model_path_list[i])
#         model2 = ViTImageClassifier.load_from_checkpoint(model_path_list[j])
#         model_list.append(model1)
#         model_list.append(model2)
#         # print(model1.state_dict().keys())
#         # print(model1.model.vit.state_dict().keys())
#         # print(model1.model.classifier.state_dict().keys())
#         # import pdb; pdb.set_trace()
#         ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model1.model.vit.state_dict()))
#         ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model2.model.vit.state_dict()))
#         # taskvector
#         sum_taskvector = sum(ckpt_list)
#         for coef in range(1, 10, 1):
#             coef = coef /10
#             merge_ckpt = sum_taskvector.apply_to(pre_model.model.vit.state_dict(), scaling_coef = coef)
#             merged_model = vit_merged_model(model_list, False)
#             merged_model.load_weight(merge_ckpt)
#             # evaluate
#             acc_upstream, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False, processor=processor)
#             print(acc_upstream, acc_merged)
#             headers=["datasets", "lambda", "acc_upstream", "acc_merged"]
#             data={"datasets":model_type + str(dataset_list), "acc_upstream":acc_upstream, "acc_merged":acc_merged, "lambda": coef}
#             result_path = '../results/new_merge_2/vit/ta.csv'
#             write_to_csv(result_path, headers, data)



# model_path_list = [
#                     '/data4/xxx/0407/ftnew/hf_vit/gtsrb/vit-large-patch32-224-in21k/version_1/checkpoints/epoch=9-step=13320.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-large-patch32-224-in21k/version_0/checkpoints/epoch=9-step=12500.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/svhn/vit-large-patch32-224-in21k/version_0/checkpoints/epoch=9-step=18320.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/sun397/vit-large-patch32-224-in21k/version_0/checkpoints/epoch=29-step=61200.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/dtd/vit-large-patch32-224-in21k/version_2/checkpoints/epoch=69-step=4130.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/cifar100/vit-large-patch32-224-in21k/version_2/checkpoints/epoch=9-step=12500.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/stanfordcars/vit-large-patch32-224-in21k/version_0/checkpoints/epoch=30_step=12648_val_loss=1.261711.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/eurosat/vit-large-patch32-224-in21k/version_0/checkpoints/epoch=9-step=5070.ckpt',
#                     '/data4/xxx/0407/ftnew/hf_vit/mnist/vit-large-patch32-224-in21k/version_3/checkpoints/epoch=9-step=30000.ckpt'
#                   ] 
# datasets_list = ['gtsrb', 'cifar10', 'svhn', 'sun397', 'dtd', 'cifar100', 'stanfordcars', 'eurosat', 'mnist']
# model_type = 'vitl32'
# model_processor_path = '/data4/xxx/0407/models/hf_vit/vit-large-patch32-224-in21k'
# pre_model = torch.load('/data4/xxx/0407/ftnew/hf_vit/cifar10/vit-large-patch32-224-in21k_pretrain.pth')
# processor = ViTImageProcessor.from_pretrained(model_processor_path + '/processor')
# # 两两分配
# for i in range(len(model_path_list)):
#     # model1 = ViTImageClassifier.load_from_checkpoint(model_path_list[i])
#     # print('ok')
#     # continue
#     for j in range(i+1, len(model_path_list), 1):
#         model_list = []  
#         ckpt_list = []
#         dataset_list = []
#         dataset_list.append(datasets_list[i])
#         dataset_list.append(datasets_list[j])
#         model1 = ViTImageClassifier.load_from_checkpoint(model_path_list[i])
#         model2 = ViTImageClassifier.load_from_checkpoint(model_path_list[j])
#         model_list.append(model1)
#         model_list.append(model2)
#         # print(model1.state_dict().keys())
#         # print(model1.model.vit.state_dict().keys())
#         # print(model1.model.classifier.state_dict().keys())
#         # import pdb; pdb.set_trace()
#         ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model1.model.vit.state_dict()))
#         ckpt_list.append(TaskVector(pre_model.model.vit.state_dict(), model2.model.vit.state_dict()))
#         # taskvector
#         sum_taskvector = sum(ckpt_list)
#         for coef in range(1, 10, 1):
#             coef = coef /10
#             merge_ckpt = sum_taskvector.apply_to(pre_model.model.vit.state_dict(), scaling_coef = coef)
#             merged_model = vit_merged_model(model_list, False)
#             merged_model.load_weight(merge_ckpt)
#             # evaluate
#             acc_upstream, acc_merged = evaluate(model_list, merged_model, dataset_list, batch_size=32, num_workers=4, normalize=False, processor=processor)
#             print(acc_upstream, acc_merged)
#             headers=["datasets", "lambda", "acc_upstream", "acc_merged"]
#             data={"datasets":model_type + str(dataset_list), "acc_upstream":acc_upstream, "acc_merged":acc_merged, "lambda": coef}
#             result_path = '../results/new_merge_2/vit/ta.csv'
#             write_to_csv(result_path, headers, data)