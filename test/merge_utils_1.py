from typing import List
import torch
import torchmetrics
from tqdm import tqdm
import torch.nn.functional as F
import sys
sys.path.append('..')
sys.path.append('../..')
from data.data_utils_1 import get_loader, get_loader_nlp
from torch.utils.data import ConcatDataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combine_loader(dataset1, dataset2):
    train_loader1, _, _ = get_loader(dataset1, 32, 1, 4, False)
    train_loader2, _, _ = get_loader(dataset2, 32, 1, 4, False)
    combine_dataset = ConcatDataset([train_loader1.dataset, train_loader2.dataset])
    combined_loader = DataLoader(
        combine_dataset,
        batch_size=128,        # 保持与原 loader 相同的 batch_size
        shuffle=True,          # 通常需要打乱
        num_workers=4,         # 与原 loader 相同的 workers
        pin_memory=True       # 与原 loader 相同的配置
    )
    return combined_loader

def combine_loader_multi(dataset_list):
    train_loader_list = [get_loader(dataset, 32, 1, 4, False)[0] for dataset in dataset_list]

    combine_dataset = ConcatDataset([loader.dataset for loader in train_loader_list])
    combined_loader = DataLoader(
        combine_dataset,
        batch_size=128,        
        shuffle=True,         
        num_workers=4,
        pin_memory=True       
    )
    return combined_loader


def reset_bn_stats(model, loader, device, number = 500):
    """Reset batch norm stats if nn.BatchNorm2d present in the model."""
    # device = get_device(model)
    # print(model)
    reset=True
    has_bn = False
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        
        if type(m) == torch.nn.BatchNorm2d:
            # print(m)
            if reset:
                m.momentum = None # use simple average
                m.reset_running_stats()
            has_bn = True

    if not has_bn:
        return model

    # run a single train epoch with augmentations to recalc stats
    model.train()
    i= 0
    with torch.no_grad():
        for images, _ in tqdm(loader, desc='Resetting BatchNorm'):
            if(i>number):
                break
            _ = model(images.to(device))
            i += 1
    return model

    
def are_elements_equal(dataset_list: List):
    # 根据datalist判断合并类型
    return len(set(dataset_list)) == 1


def evaluate_upstream_model(model, dataset, batch_size, num_workers, normalize, processor=None):
    _, test_loader, _= get_loader(dataset, batch_size, 1.0, num_workers, normalize, processor)
    model.eval()
    correct = 0
    total = 0
    model.to(device)
    # print(model)
    with torch.no_grad():
        for images, label in tqdm(test_loader, desc="testing"):
            # print(images.shape, label.shape)
            images = images.to(device)
            label = label.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            # outputs = model(images,apply_fc=True)
            # print(outputs.data)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            # break

    accuracy = (correct *1.0) / total * 100
    # print(f'Accuracy of the model on the validation set: {accuracy:.2f}%')
    return accuracy

def calculate_accuracy(outputs, true_labels, idx):
    """
    outputs: 一个包含 n 个输出头的列表，每个输出头的形状为 [batchsize, num_classes_i]
    true_labels: [batchsize], 值为 0 到 num_classes_o1-1，表示每张图片的真实类别
    idx: 表示第几个数据集
    """
    num_outputs = len(outputs)
    current_output = outputs[idx] # [batch_size, num_class]
    current_preds = torch.argmax(current_output, dim=1) # [batchsize]
    max_scores = [torch.max(output, dim=1).values for output in outputs] # [num_outputs, batchsize]
    current_correct = (current_preds == true_labels) # [batchsize]
    current_dominant = torch.all(torch.stack([max_scores[idx] >= max_scores[i] for i in range(num_outputs) if i != idx]), dim=0)
    correct = current_correct & current_dominant
    num_true = torch.sum(correct).item()


    batch_size = true_labels.size(0)

    # 计算归一化熵矩阵：[batch_size, num_heads]
    entropy_matrix = torch.stack([
        (-p.softmax(dim=1) * p.softmax(dim=1).clamp(min=1e-12).log()).sum(dim=1) / torch.log(torch.tensor(p.size(1)))
        for p in outputs
    ], dim=1)
    # 最自信头索引：[batch_size]
    best_head_indices = torch.argmin(entropy_matrix, dim=1)
    # 所有头的预测类别：[batch_size, num_heads]
    preds_tensor = torch.stack([p.argmax(dim=1) for p in outputs], dim=1)
    # 第 idx 个头的预测结果（用于与 label 比较）
    chosen_preds = preds_tensor[torch.arange(batch_size), idx]

    # 满足条件：最自信头 == idx 且预测正确
    mask = (best_head_indices == idx)
    correct = (chosen_preds == true_labels) & mask

    # 统计最终正确的样本数
    correct_count = correct.sum().item()

    return num_true, correct_count


def evaluate_merged_model(merged_model, dataset_list, batch_size, num_workers, normalize, processor=None):
    single_task=are_elements_equal(dataset_list)
    acc_merged_list=[]
    if (single_task):
        acc = evaluate_upstream_model(merged_model, dataset_list[0], batch_size, num_workers, normalize, processor)
        acc_merged_list.append(acc)
    
    else :
        joint_correct = 0
        joint_correct_entropy = 0
        total = 0
        for i, dataset in enumerate(dataset_list):
            correct_current = 0
            total_current = 0
            correct_entropy_current = 0
            _, test_loader, _ = get_loader(dataset, batch_size, 1.0, num_workers, normalize, processor)
            # print(dataset, processor)
            # import pdb; pdb.set_trace()
            merged_model.eval()
            merged_model.to(device)
            with torch.no_grad():
                for images, label in tqdm(test_loader, desc="testing"):
                    # print(images.shape, label.shape)
                    images = images.to(device)
                    # print(images)
                    label = label.to(device)
                    outputs = merged_model(images)
                    # print(outputs)
                    outputs = [F.softmax(output, dim=1) for output in outputs]
                    # print(outputs)
                    # 单个数据集-分类头评估
                    _, predicted = torch.max(outputs[i], 1)
                    total_current += label.size(0)
                    correct_current += (predicted == label).sum().item()
                    # 多个数据集-分类头输出评估
                    acc_tuple = calculate_accuracy(outputs, label, i)
                    joint_correct += acc_tuple[0] 
                    joint_correct_entropy += acc_tuple[1]
                    correct_entropy_current += acc_tuple[1]
                    total += label.size(0)
            acc_merged_list.append((correct_current * 1.0 / total_current) * 100)
            acc_merged_list.append(correct_entropy_current * 1.0 / total_current * 100)
        joint_acc = (joint_correct * 1.0 / total) * 100
        joint_entropy_acc = (joint_correct_entropy * 1.0 / total) * 100
        # assert len(acc_merged_list) == 2
        average_acc = (acc_merged_list[0] + acc_merged_list[2]) / 2
        average_entropy_acc = (acc_merged_list[1] + acc_merged_list[3]) / 2
        acc_merged_list.append((average_acc, average_entropy_acc, joint_acc, joint_entropy_acc))
    
    return acc_merged_list




def evaluate(model_list, merged_model, dataset_list, batch_size, num_workers, normalize, processor=None):
    # evaluate 上游模型
    acc_upstream_list=[]
    # print(batch_size)
    # for model, dataset in zip(model_list, dataset_list):
    #     # print(model.state_dict()['model.fc.bias'])
    #     # import pdb; pdb.set_trace()
    #     print(dataset)
    #     acc = evaluate_upstream_model(model, dataset, batch_size, num_workers, normalize, processor)
    #     print(acc)
    #     acc_upstream_list.append(acc)
        
    # evaluate merged_model
    # todo: reset bn/ln
    # acc_merged_no_resetbn_list = evaluate_merged_model(merged_model, dataset_list, batch_size, num_workers, normalize)
    # for dataset in dataset_list:
    #     train_loader, _, _ = get_loader(dataset, batch_size, 0.5, num_workers, normalize)
    #     reset_bn_stats(merged_model, train_loader, device)
    #     break
    # acc_merged__resetbn_list = evaluate_merged_model(merged_model, dataset_list, batch_size, num_workers, normalize)
    # return acc_upstream_list, [acc_merged_no_resetbn_list, acc_merged__resetbn_list]

    acc_merged_list = evaluate_merged_model(merged_model, dataset_list, batch_size, num_workers, normalize, processor)
    return acc_upstream_list, acc_merged_list



def evaluate_upstream_model_nlp(model, dataset, batch_size, num_workers, tokenizer):
    # dataset_name, batch_size, percent, num_workers, tokenizer
    _, test_loader, _= get_loader_nlp(dataset, batch_size, 1.0, num_workers, tokenizer)
    model.eval()
    correct = 0
    total = 0
    model.to(device)
    # f1
    if(dataset != 'mnli'):
        f1 = torchmetrics.classification.BinaryF1Score().to(device)
        # matthewscoef
        mcc = torchmetrics.classification.BinaryMatthewsCorrCoef().to(device)


    # print(model)
    # input_ids, attention_mask, labels = test_batch['input_ids'], test_batch['attention_mask'], test_batch['label']  # 修改为传入 input_ids 和 attention_mask
    # logits = self.forward(input_ids, attention_mask)
    with torch.no_grad():
        # print(test_loader)
        for batch in tqdm(test_loader, desc="Evaluating"):  # batch 是字典
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if(dataset != 'mnli'):
                # f1
                f1.update(predicted, labels)
                # mcc
                mcc.update(predicted, labels)
            # break

    accuracy = (correct *1.0) / total * 100
    if(dataset != 'mnli'):
        f1 = f1.compute().item()
        mcc = mcc.compute().item()
    # print(f'Accuracy of the model on the validation set: {accuracy:.2f}%')
        return accuracy, f1, mcc
    else:
        return accuracy


def evaluate_merged_model_nlp(merged_model, dataset_list, batch_size, num_workers, tokenizer):
    single_task=are_elements_equal(dataset_list)
    acc_merged_list=[]
    if (single_task):
        acc = evaluate_upstream_model_nlp(merged_model, dataset_list[0], batch_size, num_workers, tokenizer)
        acc_merged_list.append(acc)
    
    else :
        for i, dataset in enumerate(dataset_list):
            # f1
            if(dataset != 'mnli'):
                f1 = torchmetrics.classification.BinaryF1Score().to(device)
                # matthewscoef
                mcc = torchmetrics.classification.BinaryMatthewsCorrCoef().to(device)
            correct_current = 0
            total_current = 0
            _, test_loader, _ = get_loader_nlp(dataset, batch_size, 1.0, num_workers, tokenizer)
            merged_model.eval()
            merged_model.to(device)
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating"):  # batch 是字典
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    label = batch['label'].to(device)
                    outputs = merged_model(input_ids, attention_mask)                
                    # print(outputs, outputs[0].shape, outputs[1].shape)
                    outputs = [F.softmax(output, dim=1) for output in outputs]
                    # print(outputs)
                    # 单个数据集-分类头评估
                    # acc
                    _, predicted = torch.max(outputs[i], 1)
                    total_current += label.size(0)
                    correct_current += (predicted == label).sum().item()
                    if(dataset != 'mnli'):
                        # f1
                        f1.update(predicted, label)
                        # mcc
                        mcc.update(predicted, label)
                    # 多个数据集-分类头输出评估
                    
            acc_merged_list.append((correct_current * 1.0 / total_current) * 100)
            if(dataset != 'mnli'):
                acc_merged_list.append(f1.compute().item())
                acc_merged_list.append(mcc.compute().item())
        # if('mnli' not in dataset_list):
        #     assert len(acc_merged_list) == 6
        #     average_acc = (acc_merged_list[0] + acc_merged_list[3]) / 2
        #     average_f1  = (acc_merged_list[1] + acc_merged_list[4]) / 2
        #     average_mcc = (acc_merged_list[2] + acc_merged_list[5]) / 2
        #     acc_merged_list.append((average_acc, average_f1, average_mcc))
        # else :
        #     assert len(acc_merged_list) == 2
        #     average_acc = (acc_merged_list[0] + acc_merged_list[1]) / 2
        #     acc_merged_list.append(average_acc)
    
    return acc_merged_list











def evaluate_nlp(model_list, merged_model, dataset_list, batch_size, num_workers, tokenizer):
    # evaluate 上游模型
    acc_upstream_list=[]
    # print(batch_size)
    # for model, dataset in zip(model_list, dataset_list):
    #     # print(model.state_dict()['model.fc.bias'])
    #     # import pdb; pdb.set_trace()
    #     print(dataset)
    #     acc = evaluate_upstream_model(model, dataset, batch_size, num_workers, normalize, processor)
    #     print(acc)
    #     acc_upstream_list.append(acc)
        
    # evaluate merged_model
    # todo: reset bn/ln
    # acc_merged_no_resetbn_list = evaluate_merged_model(merged_model, dataset_list, batch_size, num_workers, normalize)
    # for dataset in dataset_list:
    #     train_loader, _, _ = get_loader(dataset, batch_size, 0.5, num_workers, normalize)
    #     reset_bn_stats(merged_model, train_loader, device)
    #     break
    # acc_merged__resetbn_list = evaluate_merged_model(merged_model, dataset_list, batch_size, num_workers, normalize)
    # return acc_upstream_list, [acc_merged_no_resetbn_list, acc_merged__resetbn_list]

    acc_merged_list = evaluate_merged_model_nlp(merged_model, dataset_list, batch_size, num_workers, tokenizer)
    return acc_upstream_list, acc_merged_list



def transform(checkpoint, model_name):
    if('resnet' in model_name):
        new_checkpoint = {key.replace('backbone.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        new_checkpoint = {key.replace('head.', '', 1): value for key, value in new_checkpoint.items()}
        return new_checkpoint
    if(model_name == 'vgg11'):
        old_keys = [
            'features.0.conv.weight', 
            'features.0.conv.bias', 
            'features.2.conv.weight', 
            'features.2.conv.bias', 
            'features.4.conv.weight', 
            'features.4.conv.bias', 
            'features.5.conv.weight', 
            'features.5.conv.bias', 
            'features.7.conv.weight', 
            'features.7.conv.bias', 
            'features.8.conv.weight', 
            'features.8.conv.bias', 
            'features.10.conv.weight', 
            'features.10.conv.bias', 
            'features.11.conv.weight', 
            'features.11.conv.bias', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        new_keys = [
            'features.0.weight', 
            'features.0.bias', 
            'features.3.weight', 
            'features.3.bias', 
            'features.6.weight', 
            'features.6.bias', 
            'features.8.weight', 
            'features.8.bias', 
            'features.11.weight', 
            'features.11.bias', 
            'features.13.weight', 
            'features.13.bias', 
            'features.16.weight', 
            'features.16.bias', 
            'features.18.weight', 
            'features.18.bias', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        checkpoint = {key.replace('backbone.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        for old_key, new_key in zip(old_keys, new_keys):
            checkpoint[new_key] = checkpoint.pop(old_key)
        return checkpoint 
    if(model_name == 'vgg11_bn'):
        old_keys = [
            'features.0.conv.weight', 
            'features.0.conv.bias', 
            'features.0.bn.weight', 
            'features.0.bn.bias', 
            'features.0.bn.running_mean', 
            'features.0.bn.running_var', 
            'features.0.bn.num_batches_tracked', 
            'features.2.conv.weight', 
            'features.2.conv.bias', 
            'features.2.bn.weight', 
            'features.2.bn.bias', 
            'features.2.bn.running_mean', 
            'features.2.bn.running_var', 
            'features.2.bn.num_batches_tracked', 
            'features.4.conv.weight', 
            'features.4.conv.bias', 
            'features.4.bn.weight', 
            'features.4.bn.bias', 
            'features.4.bn.running_mean', 
            'features.4.bn.running_var', 
            'features.4.bn.num_batches_tracked', 
            'features.5.conv.weight', 
            'features.5.conv.bias', 
            'features.5.bn.weight', 
            'features.5.bn.bias', 
            'features.5.bn.running_mean', 
            'features.5.bn.running_var', 
            'features.5.bn.num_batches_tracked', 
            'features.7.conv.weight', 
            'features.7.conv.bias', 
            'features.7.bn.weight', 
            'features.7.bn.bias', 
            'features.7.bn.running_mean', 
            'features.7.bn.running_var', 
            'features.7.bn.num_batches_tracked', 
            'features.8.conv.weight', 
            'features.8.conv.bias', 
            'features.8.bn.weight', 
            'features.8.bn.bias', 
            'features.8.bn.running_mean', 
            'features.8.bn.running_var', 
            'features.8.bn.num_batches_tracked', 
            'features.10.conv.weight', 
            'features.10.conv.bias', 
            'features.10.bn.weight', 
            'features.10.bn.bias', 
            'features.10.bn.running_mean', 
            'features.10.bn.running_var', 
            'features.10.bn.num_batches_tracked', 
            'features.11.conv.weight', 
            'features.11.conv.bias', 
            'features.11.bn.weight', 
            'features.11.bn.bias', 
            'features.11.bn.running_mean', 
            'features.11.bn.running_var', 
            'features.11.bn.num_batches_tracked', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        new_keys = [
        'features.0.weight', 
        'features.0.bias', 
        'features.1.weight', 
        'features.1.bias', 
        'features.1.running_mean', 
        'features.1.running_var', 
        'features.1.num_batches_tracked', 
        'features.4.weight', 
        'features.4.bias', 
        'features.5.weight', 
        'features.5.bias', 
        'features.5.running_mean', 
        'features.5.running_var', 
        'features.5.num_batches_tracked', 
        'features.8.weight', 
        'features.8.bias', 
        'features.9.weight', 
        'features.9.bias', 
        'features.9.running_mean', 
        'features.9.running_var', 
        'features.9.num_batches_tracked', 
        'features.11.weight', 
        'features.11.bias', 
        'features.12.weight', 
        'features.12.bias', 
        'features.12.running_mean', 
        'features.12.running_var', 
        'features.12.num_batches_tracked', 
        'features.15.weight', 
        'features.15.bias', 
        'features.16.weight', 
        'features.16.bias', 
        'features.16.running_mean', 
        'features.16.running_var', 
        'features.16.num_batches_tracked', 
        'features.18.weight', 
        'features.18.bias', 
        'features.19.weight', 
        'features.19.bias', 
        'features.19.running_mean', 
        'features.19.running_var', 
        'features.19.num_batches_tracked', 
        'features.22.weight', 
        'features.22.bias', 
        'features.23.weight', 
        'features.23.bias', 
        'features.23.running_mean', 
        'features.23.running_var', 
        'features.23.num_batches_tracked', 
        'features.25.weight', 
        'features.25.bias', 
        'features.26.weight', 
        'features.26.bias', 
        'features.26.running_mean', 
        'features.26.running_var', 
        'features.26.num_batches_tracked', 
        'classifier.0.weight', 
        'classifier.0.bias', 
        'classifier.3.weight', 
        'classifier.3.bias', 
        'classifier.6.weight', 
        'classifier.6.bias'
    ]
        checkpoint = {key.replace('backbone.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        for old_key, new_key in zip(old_keys, new_keys):
            checkpoint[new_key] = checkpoint.pop(old_key)
        return checkpoint
    if(model_name == 'vgg13'):
        checkpoint = {key.replace('backbone.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        old_keys = [
            'features.0.conv.weight', 
            'features.0.conv.bias', 
            'features.1.conv.weight', 
            'features.1.conv.bias', 
            'features.3.conv.weight', 
            'features.3.conv.bias', 
            'features.4.conv.weight', 
            'features.4.conv.bias', 
            'features.6.conv.weight', 
            'features.6.conv.bias', 
            'features.7.conv.weight', 
            'features.7.conv.bias', 
            'features.9.conv.weight', 
            'features.9.conv.bias', 
            'features.10.conv.weight', 
            'features.10.conv.bias', 
            'features.12.conv.weight', 
            'features.12.conv.bias', 
            'features.13.conv.weight', 
            'features.13.conv.bias', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        new_keys = [
            'features.0.weight', 
            'features.0.bias', 
            'features.2.weight', 
            'features.2.bias', 
            'features.5.weight', 
            'features.5.bias', 
            'features.7.weight', 
            'features.7.bias', 
            'features.10.weight', 
            'features.10.bias', 
            'features.12.weight', 
            'features.12.bias', 
            'features.15.weight', 
            'features.15.bias', 
            'features.17.weight', 
            'features.17.bias', 
            'features.20.weight', 
            'features.20.bias', 
            'features.22.weight', 
            'features.22.bias', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        for old_key, new_key in zip(old_keys, new_keys):
            checkpoint[new_key] = checkpoint.pop(old_key)
        return checkpoint
    if(model_name == 'vgg13_bn'):
        new_keys = [
            'features.0.weight', 
            'features.0.bias', 
            'features.1.weight', 
            'features.1.bias', 
            'features.1.running_mean', 
            'features.1.running_var', 
            'features.1.num_batches_tracked', 
            'features.3.weight', 
            'features.3.bias', 
            'features.4.weight', 
            'features.4.bias', 
            'features.4.running_mean', 
            'features.4.running_var', 
            'features.4.num_batches_tracked', 
            'features.7.weight', 
            'features.7.bias', 
            'features.8.weight', 
            'features.8.bias', 
            'features.8.running_mean', 
            'features.8.running_var', 
            'features.8.num_batches_tracked', 
            'features.10.weight', 
            'features.10.bias', 
            'features.11.weight', 
            'features.11.bias', 
            'features.11.running_mean', 
            'features.11.running_var', 
            'features.11.num_batches_tracked', 
            'features.14.weight', 
            'features.14.bias', 
            'features.15.weight', 
            'features.15.bias', 
            'features.15.running_mean', 
            'features.15.running_var', 
            'features.15.num_batches_tracked', 
            'features.17.weight', 
            'features.17.bias', 
            'features.18.weight', 
            'features.18.bias', 
            'features.18.running_mean', 
            'features.18.running_var', 
            'features.18.num_batches_tracked', 
            'features.21.weight', 
            'features.21.bias', 
            'features.22.weight', 
            'features.22.bias', 
            'features.22.running_mean', 
            'features.22.running_var', 
            'features.22.num_batches_tracked', 
            'features.24.weight', 
            'features.24.bias', 
            'features.25.weight', 
            'features.25.bias', 
            'features.25.running_mean', 
            'features.25.running_var', 
            'features.25.num_batches_tracked', 
            'features.28.weight', 
            'features.28.bias', 
            'features.29.weight', 
            'features.29.bias', 
            'features.29.running_mean', 
            'features.29.running_var', 
            'features.29.num_batches_tracked', 
            'features.31.weight', 
            'features.31.bias', 
            'features.32.weight', 
            'features.32.bias', 
            'features.32.running_mean', 
            'features.32.running_var', 
            'features.32.num_batches_tracked', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        checkpoint = {key.replace('backbone.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        old_keys = [
            'features.0.conv.weight', 
            'features.0.conv.bias', 
            'features.0.bn.weight', 
            'features.0.bn.bias', 
            'features.0.bn.running_mean', 
            'features.0.bn.running_var', 
            'features.0.bn.num_batches_tracked', 
            'features.1.conv.weight', 
            'features.1.conv.bias', 
            'features.1.bn.weight', 
            'features.1.bn.bias', 
            'features.1.bn.running_mean', 
            'features.1.bn.running_var', 
            'features.1.bn.num_batches_tracked', 
            'features.3.conv.weight', 
            'features.3.conv.bias', 
            'features.3.bn.weight', 
            'features.3.bn.bias', 
            'features.3.bn.running_mean', 
            'features.3.bn.running_var', 
            'features.3.bn.num_batches_tracked', 
            'features.4.conv.weight', 
            'features.4.conv.bias', 
            'features.4.bn.weight', 
            'features.4.bn.bias', 
            'features.4.bn.running_mean', 
            'features.4.bn.running_var', 
            'features.4.bn.num_batches_tracked', 
            'features.6.conv.weight', 
            'features.6.conv.bias', 
            'features.6.bn.weight', 
            'features.6.bn.bias', 
            'features.6.bn.running_mean', 
            'features.6.bn.running_var', 
            'features.6.bn.num_batches_tracked', 
            'features.7.conv.weight', 
            'features.7.conv.bias', 
            'features.7.bn.weight', 
            'features.7.bn.bias', 
            'features.7.bn.running_mean', 
            'features.7.bn.running_var', 
            'features.7.bn.num_batches_tracked', 
            'features.9.conv.weight', 
            'features.9.conv.bias', 
            'features.9.bn.weight', 
            'features.9.bn.bias', 
            'features.9.bn.running_mean', 
            'features.9.bn.running_var', 
            'features.9.bn.num_batches_tracked', 
            'features.10.conv.weight', 
            'features.10.conv.bias', 
            'features.10.bn.weight', 
            'features.10.bn.bias', 
            'features.10.bn.running_mean', 
            'features.10.bn.running_var', 
            'features.10.bn.num_batches_tracked', 
            'features.12.conv.weight', 
            'features.12.conv.bias', 
            'features.12.bn.weight', 
            'features.12.bn.bias', 
            'features.12.bn.running_mean', 
            'features.12.bn.running_var', 
            'features.12.bn.num_batches_tracked', 
            'features.13.conv.weight', 
            'features.13.conv.bias', 
            'features.13.bn.weight', 
            'features.13.bn.bias', 
            'features.13.bn.running_mean', 
            'features.13.bn.running_var', 
            'features.13.bn.num_batches_tracked', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        for old_key, new_key in zip(old_keys, new_keys):
            checkpoint[new_key] = checkpoint.pop(old_key)
        return checkpoint
    if(model_name == 'vgg16'):
        checkpoint = {key.replace('backbone.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        old_keys = [
            'features.0.conv.weight', 
            'features.0.conv.bias', 
            'features.1.conv.weight', 
            'features.1.conv.bias', 
            'features.3.conv.weight', 
            'features.3.conv.bias', 
            'features.4.conv.weight', 
            'features.4.conv.bias', 
            'features.6.conv.weight', 
            'features.6.conv.bias', 
            'features.7.conv.weight', 
            'features.7.conv.bias', 
            'features.8.conv.weight', 
            'features.8.conv.bias', 
            'features.10.conv.weight', 
            'features.10.conv.bias', 
            'features.11.conv.weight', 
            'features.11.conv.bias', 
            'features.12.conv.weight', 
            'features.12.conv.bias', 
            'features.14.conv.weight', 
            'features.14.conv.bias', 
            'features.15.conv.weight', 
            'features.15.conv.bias', 
            'features.16.conv.weight', 
            'features.16.conv.bias', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        new_keys = [
        'features.0.weight', 
        'features.0.bias', 
        'features.2.weight', 
        'features.2.bias', 
        'features.5.weight', 
        'features.5.bias', 
        'features.7.weight', 
        'features.7.bias', 
        'features.10.weight', 
        'features.10.bias', 
        'features.12.weight', 
        'features.12.bias', 
        'features.14.weight', 
        'features.14.bias', 
        'features.17.weight', 
        'features.17.bias', 
        'features.19.weight', 
        'features.19.bias', 
        'features.21.weight', 
        'features.21.bias', 
        'features.24.weight', 
        'features.24.bias', 
        'features.26.weight', 
        'features.26.bias', 
        'features.28.weight', 
        'features.28.bias', 
        'classifier.0.weight', 
        'classifier.0.bias', 
        'classifier.3.weight', 
        'classifier.3.bias', 
        'classifier.6.weight', 
        'classifier.6.bias',
        ]
        for old_key, new_key in zip(old_keys, new_keys):
            checkpoint[new_key] = checkpoint.pop(old_key)
        return checkpoint
    if(model_name == 'vgg16_bn'):
        checkpoint = {key.replace('backbone.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        old_keys = [
            'features.0.conv.weight', 
            'features.0.conv.bias', 
            'features.0.bn.weight', 
            'features.0.bn.bias', 
            'features.0.bn.running_mean', 
            'features.0.bn.running_var', 
            'features.0.bn.num_batches_tracked', 
            'features.1.conv.weight', 
            'features.1.conv.bias', 
            'features.1.bn.weight', 
            'features.1.bn.bias', 
            'features.1.bn.running_mean', 
            'features.1.bn.running_var', 
            'features.1.bn.num_batches_tracked', 
            'features.3.conv.weight', 
            'features.3.conv.bias', 
            'features.3.bn.weight', 
            'features.3.bn.bias', 
            'features.3.bn.running_mean', 
            'features.3.bn.running_var', 
            'features.3.bn.num_batches_tracked', 
            'features.4.conv.weight', 
            'features.4.conv.bias', 
            'features.4.bn.weight', 
            'features.4.bn.bias', 
            'features.4.bn.running_mean', 
            'features.4.bn.running_var', 
            'features.4.bn.num_batches_tracked', 
            'features.6.conv.weight', 
            'features.6.conv.bias', 
            'features.6.bn.weight', 
            'features.6.bn.bias', 
            'features.6.bn.running_mean', 
            'features.6.bn.running_var', 
            'features.6.bn.num_batches_tracked', 
            'features.7.conv.weight', 
            'features.7.conv.bias', 
            'features.7.bn.weight', 
            'features.7.bn.bias', 
            'features.7.bn.running_mean', 
            'features.7.bn.running_var', 
            'features.7.bn.num_batches_tracked', 
            'features.8.conv.weight', 
            'features.8.conv.bias', 
            'features.8.bn.weight', 
            'features.8.bn.bias', 
            'features.8.bn.running_mean', 
            'features.8.bn.running_var', 
            'features.8.bn.num_batches_tracked', 
            'features.10.conv.weight', 
            'features.10.conv.bias', 
            'features.10.bn.weight', 
            'features.10.bn.bias', 
            'features.10.bn.running_mean', 
            'features.10.bn.running_var', 
            'features.10.bn.num_batches_tracked', 
            'features.11.conv.weight', 
            'features.11.conv.bias', 
            'features.11.bn.weight', 
            'features.11.bn.bias', 
            'features.11.bn.running_mean', 
            'features.11.bn.running_var', 
            'features.11.bn.num_batches_tracked', 
            'features.12.conv.weight', 
            'features.12.conv.bias', 
            'features.12.bn.weight', 
            'features.12.bn.bias', 
            'features.12.bn.running_mean', 
            'features.12.bn.running_var', 
            'features.12.bn.num_batches_tracked', 
            'features.14.conv.weight', 
            'features.14.conv.bias', 
            'features.14.bn.weight', 
            'features.14.bn.bias', 
            'features.14.bn.running_mean', 
            'features.14.bn.running_var', 
            'features.14.bn.num_batches_tracked', 
            'features.15.conv.weight', 
            'features.15.conv.bias', 
            'features.15.bn.weight', 
            'features.15.bn.bias', 
            'features.15.bn.running_mean', 
            'features.15.bn.running_var', 
            'features.15.bn.num_batches_tracked', 
            'features.16.conv.weight', 
            'features.16.conv.bias', 
            'features.16.bn.weight', 
            'features.16.bn.bias', 
            'features.16.bn.running_mean', 
            'features.16.bn.running_var', 
            'features.16.bn.num_batches_tracked', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        new_keys = [
            'features.0.weight', 
            'features.0.bias', 
            'features.1.weight', 
            'features.1.bias', 
            'features.1.running_mean', 
            'features.1.running_var', 
            'features.1.num_batches_tracked', 
            'features.3.weight', 
            'features.3.bias', 
            'features.4.weight', 
            'features.4.bias', 
            'features.4.running_mean', 
            'features.4.running_var', 
            'features.4.num_batches_tracked', 
            'features.7.weight', 
            'features.7.bias', 
            'features.8.weight', 
            'features.8.bias', 
            'features.8.running_mean', 
            'features.8.running_var', 
            'features.8.num_batches_tracked', 
            'features.10.weight', 
            'features.10.bias', 
            'features.11.weight', 
            'features.11.bias', 
            'features.11.running_mean', 
            'features.11.running_var', 
            'features.11.num_batches_tracked', 
            'features.14.weight', 
            'features.14.bias', 
            'features.15.weight', 
            'features.15.bias', 
            'features.15.running_mean', 
            'features.15.running_var', 
            'features.15.num_batches_tracked', 
            'features.17.weight', 
            'features.17.bias', 
            'features.18.weight', 
            'features.18.bias', 
            'features.18.running_mean', 
            'features.18.running_var', 
            'features.18.num_batches_tracked', 
            'features.20.weight', 
            'features.20.bias', 
            'features.21.weight', 
            'features.21.bias', 
            'features.21.running_mean', 
            'features.21.running_var', 
            'features.21.num_batches_tracked', 
            'features.24.weight', 
            'features.24.bias', 
            'features.25.weight', 
            'features.25.bias', 
            'features.25.running_mean', 
            'features.25.running_var', 
            'features.25.num_batches_tracked', 
            'features.27.weight', 
            'features.27.bias', 
            'features.28.weight', 
            'features.28.bias', 
            'features.28.running_mean', 
            'features.28.running_var', 
            'features.28.num_batches_tracked', 
            'features.30.weight', 
            'features.30.bias', 
            'features.31.weight', 
            'features.31.bias', 
            'features.31.running_mean', 
            'features.31.running_var', 
            'features.31.num_batches_tracked', 
            'features.34.weight', 
            'features.34.bias', 
            'features.35.weight', 
            'features.35.bias', 
            'features.35.running_mean', 
            'features.35.running_var', 
            'features.35.num_batches_tracked', 
            'features.37.weight', 
            'features.37.bias', 
            'features.38.weight', 
            'features.38.bias', 
            'features.38.running_mean', 
            'features.38.running_var', 
            'features.38.num_batches_tracked', 
            'features.40.weight', 
            'features.40.bias', 
            'features.41.weight', 
            'features.41.bias', 
            'features.41.running_mean', 
            'features.41.running_var', 
            'features.41.num_batches_tracked', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        for old_key, new_key in zip(old_keys, new_keys):
            checkpoint[new_key] = checkpoint.pop(old_key)
        return checkpoint
    if(model_name == 'vgg19'):
        checkpoint = {key.replace('backbone.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        old_keys = [
            'features.0.conv.weight', 
            'features.0.conv.bias', 
            'features.1.conv.weight', 
            'features.1.conv.bias', 
            'features.3.conv.weight', 
            'features.3.conv.bias', 
            'features.4.conv.weight', 
            'features.4.conv.bias', 
            'features.6.conv.weight', 
            'features.6.conv.bias', 
            'features.7.conv.weight', 
            'features.7.conv.bias', 
            'features.8.conv.weight', 
            'features.8.conv.bias', 
            'features.9.conv.weight', 
            'features.9.conv.bias', 
            'features.11.conv.weight', 
            'features.11.conv.bias', 
            'features.12.conv.weight', 
            'features.12.conv.bias', 
            'features.13.conv.weight', 
            'features.13.conv.bias', 
            'features.14.conv.weight', 
            'features.14.conv.bias', 
            'features.16.conv.weight', 
            'features.16.conv.bias', 
            'features.17.conv.weight', 
            'features.17.conv.bias', 
            'features.18.conv.weight', 
            'features.18.conv.bias', 
            'features.19.conv.weight', 
            'features.19.conv.bias', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias',
        ]

        new_keys = [
            'features.0.weight', 
            'features.0.bias',
            'features.2.weight', 
            'features.2.bias', 
            'features.5.weight', 
            'features.5.bias', 
            'features.7.weight', 
            'features.7.bias', 
            'features.10.weight',
            'features.10.bias', 
            'features.12.weight', 
            'features.12.bias', 
            'features.14.weight', 
            'features.14.bias', 
            'features.16.weight', 
            'features.16.bias', 
            'features.19.weight', 
            'features.19.bias', 
            'features.21.weight', 
            'features.21.bias', 
            'features.23.weight', 
            'features.23.bias', 
            'features.25.weight', 
            'features.25.bias', 
            'features.28.weight', 
            'features.28.bias', 
            'features.30.weight', 
            'features.30.bias', 
            'features.32.weight', 
            'features.32.bias', 
            'features.34.weight', 
            'features.34.bias', 
            'classifier.0.weight', 
            'classifier.0.bias', 
            'classifier.3.weight', 
            'classifier.3.bias', 
            'classifier.6.weight', 
            'classifier.6.bias'
        ]
        for old_key, new_key in zip(old_keys, new_keys):
            checkpoint[new_key] = checkpoint.pop(old_key)
        return checkpoint
    if(model_name == 'vgg19_bn'):
        checkpoint = {key.replace('backbone.', '', 1): value for key, value in checkpoint['state_dict'].items()}
        old_keys = [
            'features.0.conv.weight', 'features.0.conv.bias', 'features.0.bn.weight', 'features.0.bn.bias', 
            'features.0.bn.running_mean', 'features.0.bn.running_var', 'features.0.bn.num_batches_tracked', 
            'features.1.conv.weight', 'features.1.conv.bias', 'features.1.bn.weight', 'features.1.bn.bias', 
            'features.1.bn.running_mean', 'features.1.bn.running_var', 'features.1.bn.num_batches_tracked',
            'features.3.conv.weight', 'features.3.conv.bias', 'features.3.bn.weight', 'features.3.bn.bias', 
            'features.3.bn.running_mean', 'features.3.bn.running_var', 'features.3.bn.num_batches_tracked',
            'features.4.conv.weight', 'features.4.conv.bias', 'features.4.bn.weight', 'features.4.bn.bias', 
            'features.4.bn.running_mean', 'features.4.bn.running_var', 'features.4.bn.num_batches_tracked',
            'features.6.conv.weight', 'features.6.conv.bias', 'features.6.bn.weight', 'features.6.bn.bias', 
            'features.6.bn.running_mean', 'features.6.bn.running_var', 'features.6.bn.num_batches_tracked',
            'features.7.conv.weight', 'features.7.conv.bias', 'features.7.bn.weight', 'features.7.bn.bias', 
            'features.7.bn.running_mean', 'features.7.bn.running_var', 'features.7.bn.num_batches_tracked',
            'features.8.conv.weight', 'features.8.conv.bias', 'features.8.bn.weight', 'features.8.bn.bias', 
            'features.8.bn.running_mean', 'features.8.bn.running_var', 'features.8.bn.num_batches_tracked',
            'features.9.conv.weight', 'features.9.conv.bias', 'features.9.bn.weight', 'features.9.bn.bias', 
            'features.9.bn.running_mean', 'features.9.bn.running_var', 'features.9.bn.num_batches_tracked',
            'features.11.conv.weight', 'features.11.conv.bias', 'features.11.bn.weight', 'features.11.bn.bias', 
            'features.11.bn.running_mean', 'features.11.bn.running_var', 'features.11.bn.num_batches_tracked',
            'features.12.conv.weight', 'features.12.conv.bias', 'features.12.bn.weight', 'features.12.bn.bias', 
            'features.12.bn.running_mean', 'features.12.bn.running_var', 'features.12.bn.num_batches_tracked',
            'features.13.conv.weight', 'features.13.conv.bias', 'features.13.bn.weight', 'features.13.bn.bias', 
            'features.13.bn.running_mean', 'features.13.bn.running_var', 'features.13.bn.num_batches_tracked',
            'features.14.conv.weight', 'features.14.conv.bias', 'features.14.bn.weight', 'features.14.bn.bias', 
            'features.14.bn.running_mean', 'features.14.bn.running_var', 'features.14.bn.num_batches_tracked',
            'features.16.conv.weight', 'features.16.conv.bias', 'features.16.bn.weight', 'features.16.bn.bias', 
            'features.16.bn.running_mean', 'features.16.bn.running_var', 'features.16.bn.num_batches_tracked',
            'features.17.conv.weight', 'features.17.conv.bias', 'features.17.bn.weight', 'features.17.bn.bias', 
            'features.17.bn.running_mean', 'features.17.bn.running_var', 'features.17.bn.num_batches_tracked',
            'features.18.conv.weight', 'features.18.conv.bias', 'features.18.bn.weight', 'features.18.bn.bias', 
            'features.18.bn.running_mean', 'features.18.bn.running_var', 'features.18.bn.num_batches_tracked',
            'features.19.conv.weight', 'features.19.conv.bias', 'features.19.bn.weight', 'features.19.bn.bias', 
            'features.19.bn.running_mean', 'features.19.bn.running_var', 'features.19.bn.num_batches_tracked'
        ]

        new_keys = [
            'features.0.weight', 'features.0.bias', 'features.1.weight', 'features.1.bias', 
            'features.1.running_mean', 'features.1.running_var', 'features.1.num_batches_tracked', 
            'features.3.weight', 'features.3.bias', 'features.4.weight', 'features.4.bias', 
            'features.4.running_mean', 'features.4.running_var', 'features.4.num_batches_tracked',
            'features.7.weight', 'features.7.bias', 'features.8.weight', 'features.8.bias', 
            'features.8.running_mean', 'features.8.running_var', 'features.8.num_batches_tracked',
            'features.10.weight', 'features.10.bias', 'features.11.weight', 'features.11.bias', 
            'features.11.running_mean', 'features.11.running_var', 'features.11.num_batches_tracked',
            'features.14.weight', 'features.14.bias', 'features.15.weight', 'features.15.bias', 
            'features.15.running_mean', 'features.15.running_var', 'features.15.num_batches_tracked',
            'features.17.weight', 'features.17.bias', 'features.18.weight', 'features.18.bias', 
            'features.18.running_mean', 'features.18.running_var', 'features.18.num_batches_tracked',
            'features.20.weight', 'features.20.bias', 'features.21.weight', 'features.21.bias', 
            'features.21.running_mean', 'features.21.running_var', 'features.21.num_batches_tracked',
            'features.23.weight', 'features.23.bias', 'features.24.weight', 'features.24.bias', 
            'features.24.running_mean', 'features.24.running_var', 'features.24.num_batches_tracked',
            'features.27.weight', 'features.27.bias', 'features.28.weight', 'features.28.bias', 
            'features.28.running_mean', 'features.28.running_var', 'features.28.num_batches_tracked',
            'features.30.weight', 'features.30.bias', 'features.31.weight', 'features.31.bias', 
            'features.31.running_mean', 'features.31.running_var', 'features.31.num_batches_tracked',
            'features.33.weight', 'features.33.bias', 'features.34.weight', 'features.34.bias', 
            'features.34.running_mean', 'features.34.running_var', 'features.34.num_batches_tracked',
            'features.36.weight', 'features.36.bias', 'features.37.weight', 'features.37.bias', 
            'features.37.running_mean', 'features.37.running_var', 'features.37.num_batches_tracked',
            'features.40.weight', 'features.40.bias', 'features.41.weight', 'features.41.bias', 
            'features.41.running_mean', 'features.41.running_var', 'features.41.num_batches_tracked',
            'features.43.weight', 'features.43.bias', 'features.44.weight', 'features.44.bias', 
            'features.44.running_mean', 'features.44.running_var', 'features.44.num_batches_tracked',
            'features.46.weight', 'features.46.bias', 'features.47.weight', 'features.47.bias', 
            'features.47.running_mean', 'features.47.running_var', 'features.47.num_batches_tracked',
            'features.49.weight', 'features.49.bias', 'features.50.weight', 'features.50.bias', 
            'features.50.running_mean', 'features.50.running_var', 'features.50.num_batches_tracked'
        ]
        for old_key, new_key in zip(old_keys, new_keys):
            checkpoint[new_key] = checkpoint.pop(old_key)
        return checkpoint
    # if(model_name == 'densenet121'):

    # if(model_name == 'densenet161'):
    # if(model_name == 'densenet169'):
    # if(model_name == 'densenet201'):    