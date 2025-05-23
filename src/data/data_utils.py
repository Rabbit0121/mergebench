import sys
from typing import Dict, List
from datasets import load_dataset, load_from_disk
import torch
import torchvision
sys.path.append('/home/xxx/miniconda3/envs/0407/SecureMerge/src/data')
from transform import train_transform_without_normalize, train_transform, test_transform_without_normalize, test_transform, mnist_test_transform_without_normalize, mnist_train_transform_without_normalize
from imagenet import *
from torch.utils.data import DataLoader, random_split
from datasets import concatenate_datasets



# from https://github.com/openai/CLIP/blob/main/data/prompts.md?plain=1


class ViTDataset(Dataset):
    def __init__(self, split='train', train=True, processor=None, dataset='cifar10'):
        self.split = split
        self.train = train
        self.dataset_name = dataset
        # print(dataset)
        if(dataset == 'cifar10'):
            self.dataset = torchvision.datasets.CIFAR10(root='/data4/xxx/dataset/cifar', train=train, download=False)
        if(dataset == 'cifar100'):
            self.dataset = torchvision.datasets.CIFAR100(root='/data4/xxx/dataset/cifar', train=train, download=False)
        if(dataset == 'dtd'):
            self.dataset = torchvision.datasets.DTD(root='/data4/xxx/dataset', split=split, download=False)
        if(dataset == 'eurosat'):
            self.dataset = torchvision.datasets.EuroSAT(root='/data4/xxx/dataset', download=False)
        if(dataset == 'mnist'):
            self.dataset = torchvision.datasets.MNIST(root='/data4/xxx/dataset', train=train, download=False)
        if(dataset == 'sun397'):
            self.dataset = torchvision.datasets.SUN397(root='/data4/xxx/dataset', download=False)
        if(dataset == 'svhn'):
            self.dataset = torchvision.datasets.SVHN(root='/data4/xxx/dataset/svhn', split=split, download=False)
        if(dataset == 'gtsrb'):
            self.dataset = torchvision.datasets.GTSRB(root='/data4/xxx/dataset', split=split, download=False)
        if(dataset == 'stanfordcars'):
            self.dataset = torchvision.datasets.StanfordCars(root='/data4/xxx/dataset', split=split, download=False)
        self.processor=processor
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.convert("RGB")
        if (self.split == 'train' or self.train == True):
            train_transform_vit = [
                # transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
            ]
            train_transform_vit = transforms.Compose(train_transform_vit)
            img = train_transform_vit(img)
        if self.processor:
            img = self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0) 
            # print(img)
        return img, label


def get_num_classes(dataset):
    dataset_classes = {"cifar10": 10, "cifar100": 100, "dtd": 47, "eurosat": 10, "gtsrb": 43, "imagenet": 1000, 
                       "mnist": 10, "stanfordcars": 196, "sun397": 397, "svhn": 10, "sst2":2, "cola":2, 
                       "mnli":3, "mrpc":2, "qnli":2, "qqp":2, "rte":2}
    return dataset_classes[dataset]


def get_loader(dataset, batch_size, percent, num_workers=16, normalize=False, processor=None):
    train_loader = None
    test_loader = None
    val_loader = None
    train_dataset = None
    val_dataset = None
    test_dataset = None
    # print(dataset, batch_size, percent, num_workers, normalize, processor)
    if normalize == False:
        train_transform_used = train_transform_without_normalize
        test_transform_used = test_transform_without_normalize
    else:
        train_transform_used = train_transform
        test_transform_used = test_transform
    if processor == None: # use transforme
        if(dataset == 'imagenet'): 
            test_dataset = ImageNet('/data4/xxx/dataset/ImageNet/val', labels, transform=test_transform_used)
            train_dataset = ImageNet('/data4/xxx/dataset/ImageNet/train_images_0', labels, transform=train_transform_used)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            # print("aaaaaaa",val_size)
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        elif(dataset == 'cifar10'):

            test_dataset = torchvision.datasets.CIFAR10(root='/data4/xxx/dataset/cifar', train=False, transform=test_transform_used, download=False)
            train_dataset = torchvision.datasets.CIFAR10(root='/data4/xxx/dataset/cifar', train=True, transform=train_transform_used, download=False)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
 
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        elif(dataset == 'cifar100'):

            test_dataset = torchvision.datasets.CIFAR100(root='/data4/xxx/dataset/cifar', train=False, transform=test_transform_used, download=False)
            train_dataset = torchvision.datasets.CIFAR100(root='/data4/xxx/dataset/cifar', train=True, transform=train_transform_used, download=False)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        elif(dataset == 'dtd'):

            test_dataset = torchvision.datasets.DTD(root='/data4/xxx/dataset', split='test', transform=test_transform_used, download=True)
            train_dataset = torchvision.datasets.DTD(root='/data4/xxx/dataset', split='train', transform=train_transform_used, download=True)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            val_dataset = torchvision.datasets.DTD(root='/data4/xxx/dataset', split='val', transform=test_transform_used, download=True)

        elif (dataset == 'eurosat'):

            train_dataset = torchvision.datasets.EuroSAT(root='/data4/xxx/dataset', transform=train_transform_used, download=False)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_dataset, test_dataset = random_split(train_dataset, [int(0.6*len(train_dataset)), len(train_dataset)-int(0.6*len(train_dataset))])
            test_dataset, val_dataset = random_split(test_dataset, [int(0.5*len(test_dataset)), len(test_dataset)-int(0.5*len(test_dataset))])

        elif (dataset == 'mnist'):

            train_transform_used = mnist_train_transform_without_normalize
            test_transform_used = mnist_test_transform_without_normalize  
            test_dataset = torchvision.datasets.MNIST(root='/data4/xxx/dataset', train=False, transform=test_transform_used, download=False)
            train_dataset = torchvision.datasets.MNIST(root='/data4/xxx/dataset', train=True, transform=train_transform_used, download=False)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        elif (dataset == 'sun397'):

            train_dataset = torchvision.datasets.SUN397(root='/data4/xxx/dataset', transform=train_transform_used, download=False)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_dataset, test_dataset = random_split(train_dataset, [int(0.6*len(train_dataset)), len(train_dataset)-int(0.6*len(train_dataset))])
            test_dataset, val_dataset = random_split(test_dataset, [int(0.5*len(test_dataset)), len(test_dataset)-int(0.5*len(test_dataset))])

        elif (dataset == 'svhn'):

            train_dataset = torchvision.datasets.SVHN(root='/data4/xxx/dataset/svhn', split='train', transform=train_transform_used, download=False)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_dataset, val_dataset = random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))])
            test_dataset = torchvision.datasets.SVHN(root='/data4/xxx/dataset/svhn', split='test', transform=test_transform_used, download=False)
    
        elif (dataset == 'gtsrb'):

            train_dataset = torchvision.datasets.GTSRB(root='/data4/xxx/dataset', split='train', transform=train_transform_used, download=False)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_dataset, val_dataset = random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))])
            test_dataset = torchvision.datasets.GTSRB(root='/data4/xxx/dataset', split='test', transform=test_transform_used, download=False)

        elif (dataset == 'stanfordcars'):
            # from：https://tanxy.club/2023/torchvision_for_stanfordcars_error
            train_dataset = torchvision.datasets.StanfordCars(root='/data4/xxx/dataset', split='train', transform=train_transform_used, download=False)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_dataset, val_dataset = random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))])
            test_dataset = torchvision.datasets.StanfordCars(root='/data4/xxx/dataset', split='test', transform=test_transform_used, download=False)
        else: raise ValueError("Undefined dataset.")
    
    else: 
        if (dataset in ['cifar10', 'cifar100', 'mnist']): # train
            train_dataset = ViTDataset(train=True, processor=processor, dataset=dataset)
            test_dataset = ViTDataset(train=False, processor=processor, dataset=dataset)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
        elif (dataset in ['dtd', 'svhn', 'gtsrb', 'stanfordcars']): # split
            train_dataset = ViTDataset(split='train', processor=processor, dataset=dataset)
            test_dataset = ViTDataset(split='test', processor=processor, dataset=dataset)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            if(dataset == 'dtd'):
                val_dataset = ViTDataset(split='val', processor=processor, dataset=dataset)
            else: train_dataset, val_dataset = random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))])

        elif (dataset in ['eurosat', 'sun397']): # all
            train_dataset = ViTDataset(processor=processor, dataset=dataset)
            train_dataset, _ = random_split(train_dataset, [int(percent*len(train_dataset)), len(train_dataset)-int(percent*len(train_dataset))])
            train_dataset, test_dataset = random_split(train_dataset, [int(0.6*len(train_dataset)), len(train_dataset)-int(0.6*len(train_dataset))])
            test_dataset, val_dataset = random_split(test_dataset, [int(0.5*len(test_dataset)), len(test_dataset)-int(0.5*len(test_dataset))])
        
        else: raise ValueError("Undefined dataset.")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if train_loader is None or val_loader is None or test_loader is None:
        raise ValueError("Failed to load dataloaders.")
    return train_loader, test_loader, val_loader



def preprocess_function(examples, tokenizer, dataset_name):
    if dataset_name in ["sst2", "cola"]:
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    elif dataset_name in ["mnli"]:
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)

    elif dataset_name in ["mrpc", "rte", "wnli"]:
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    elif dataset_name in ["qnli"]:
        return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation='longest_first')
    
    elif dataset_name in ["qqp"]:
        return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True)
    

def get_loader_nlp(dataset_name, batch_size, percent, num_workers, tokenizer, adv=False):
    if(adv == False):
        cache_dir = f"/data4/xxx/tmp"
        data_dir = "/data4/xxx/dataset/glue" + f"/{dataset_name}"
        if dataset_name in ["sst2", "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "wnli"] == False:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        try:
            dataset = load_from_disk(data_dir)
        except:
            dataset = load_dataset("glue", dataset_name, cache_dir=cache_dir)
            dataset.save_to_disk(data_dir)
        # print(dataset["train"][0])
        
        tokenized_train = dataset["train"].map(lambda x: preprocess_function(x, tokenizer, dataset_name), batched=True)
        tokenized_train.set_format(type="torch")

        train_dataset, _ = random_split(tokenized_train, [int(len(tokenized_train) * percent), len(tokenized_train) - int(len(tokenized_train) * percent)])
        if(dataset_name == 'mnli'):
            test_dataset_matched = dataset["validation_matched"]
            test_dataset_mismatched = dataset["validation_mismatched"]
            test_dataset = concatenate_datasets([test_dataset_matched, test_dataset_mismatched])
            test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer, dataset_name), batched=True)
            test_dataset.set_format(type="torch")
        else:
            test_dataset = dataset["validation"].map(lambda x: preprocess_function(x, tokenizer, dataset_name), batched=True)
            test_dataset.set_format(type="torch")


        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return train_loader, test_loader, _
    

    elif(adv == True):
        cache_dir = f"/data4/xxx/tmp"
        data_dir = "/data4/xxx/dataset/advglue" + f"/{dataset_name}"
       
        if dataset_name in ["sst2", "mnli", "qnli", "qqp", "rte", "mnli_mismatched"] == False:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        try:
            dataset = load_from_disk(data_dir)
        except:
            dataset = load_dataset("adv_glue", "adv_" + dataset_name, cache_dir=cache_dir)
            dataset.save_to_disk(data_dir)
        # print(dataset["train"][0])
        
        val_dataset = dataset["validation"].map(lambda x: preprocess_function(x, tokenizer, dataset_name), batched=True)
        val_dataset.set_format(type="torch")
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return val_loader
