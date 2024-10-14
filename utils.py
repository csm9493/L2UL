from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, Subset

from advertorch.attacks import L2PGDAttack

from torchvision import datasets, transforms

from typing import *

import copy
import itertools
from itertools import cycle
from resnet_cifar100 import resnet18, resnet34, resnet50

import numpy as np
import random

from tqdm import tqdm
import glob
from PIL import Image

#from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class UTKDataset(Dataset):
    
    def __init__(self, root, age_grouping, transform):
        self.path = root
        file_list = glob.glob(self.path + "/*.jpg")
        self.data = []

        if age_grouping == 'TNN': # grouping from https://github.com/ArminBaz/UTK-Face/tree/master
            self.bins = np.array([0,10,15,20,25,30,40,50,60,120])
        elif age_grouping == 'MFD': # grouping from https://arxiv.org/abs/2106.04411
            self.bins = np.array([0,20,41,120])
        elif age_grouping == 'balanced': # balanced grouping:
            self.bins = np.array([0,20,30,45,120])
        elif age_grouping == 'groups': # teen, adult, etc.
            self.bins = np.array([0,4,13,20,31,46,61,120])
        elif age_grouping == 'tens':
            self.bins = np.array([0,11,21,31,41,51,61,71,81,91,101,111,120])
        else:
            raise NotImplementedError
        
        self.transform = transform
        
        for f in file_list:
            age = int(f.split('_')[0].split('/')[-1])
            class_name = np.where(age < self.bins)[0][0] - 1
            self.data.append([f, class_name])
            
    def __len__(self):
        
        return len(self.data)
        
    def __getitem__(self, idx):

        f, class_name = self.data[idx]
        img = Image.open(f)
        
        if self.transform:
            img = self.transform(img)

        #img = torch.from_numpy(img)
        #img = img.permute(2,0,1).float()
        
        return img, class_name

class FeatureExtractor:
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("Ahmed9275/Vit-Cifar100")

    def __call__(self, image):
        output = self.feature_extractor(list(image.unsqueeze(0).cpu()), return_tensors="pt")
        return output['pixel_values'][0]

class ViTModel(nn.Module):
    
    # Load pretrained ViT model
    def __init__(self):
        super(ViTModel, self).__init__()
        
        # Load ViT finetuned on Cifar100 https://huggingface.co/Ahmed9275/Vit-Cifar100
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("Ahmed9275/Vit-Cifar100")
        self.encoder = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")
        
    def forward(self, x):
        x = self.encoder(pixel_values=x)
        return x.logits

class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self._len = len(inputs)

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


class NormalizeLayer(nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds

def getDataLoaders(unlearn_k: int,
                   unlearn_label: int,
                   train_dataset,
                   test_dataset,
                   naive_unlearn_kwargs,
                   test_kwargs):
                   
    train_labels = torch.from_numpy(np.array(train_dataset.targets))

    #select unlearning data
    indices_k_unlearn = torch.randperm(train_labels.shape[0])[:unlearn_k]

    copy_train_labels = train_labels.clone()
    copy_train_labels[indices_k_unlearn] = -10

    indices_other_data = (copy_train_labels != -10).nonzero(as_tuple=False)

    f_dataset = Subset(train_dataset, indices_k_unlearn.view(-1,))
    f_loader = torch.utils.data.DataLoader(f_dataset,**naive_unlearn_kwargs)

    r_dataset = Subset(train_dataset, indices_other_data.view(-1,))
    r_loader = torch.utils.data.DataLoader(r_dataset,**test_kwargs)

    t_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    print ('len(forget_dataset) : ', len(f_dataset), ' ',
           'len(residual_dataset) : ', len(r_dataset), ' ',
           'len(test_dataset) : ', len(test_dataset))
    
    return f_loader, r_loader, t_loader
    
def naive_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    CE = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = -CE(output, target)
        loss.backward()
        optimizer.step()
        
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    CE = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = CE(output, target)
        loss.backward()
        optimizer.step()
        
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    CE = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += CE(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, 100. * correct / len(test_loader.dataset)
        
def adv_attack(args, model, device, train_loader, adversary, unlearn_k, num_classes=10, num_adv_images = None, indices=None):
    model.eval()
    
    attacked_image_arr = []
    target_label_arr = []
    image_idx = []
    
    num_iters = num_adv_images

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        for i in tqdm(range(num_iters)):
            
            attack_label = torch.rand(data.shape[0]).cuda() * num_classes
            attack_label = attack_label.to(torch.long)
            attack_label = torch.where(attack_label == target, (torch.rand(data.shape[0]).long().cuda()*num_classes + num_classes) // 2, attack_label)

            adv_example = adversary.perturb(data, attack_label)

            inputs_numpy = adv_example.detach().cpu().numpy()
            labels_numpy = attack_label.cpu().numpy()

            for j in range(inputs_numpy.shape[0]):

                attacked_image_arr.append(inputs_numpy[j])
                target_label_arr.append(labels_numpy[j])
                image_idx.append(indices[j + (batch_idx * 128)].item())
                
            
    return attacked_image_arr, target_label_arr, image_idx

def estimate_parameter_importance(trn_loader, model, device, num_samples, optimizer):
    # Initialize importance matrices
    importance = {n: torch.zeros(p.shape).to(device) for n, p in model.named_parameters()
                  if p.requires_grad}
    
    # Compute fisher information for specified number of samples -- rounded to the batch size
    n_samples_batches = (num_samples // trn_loader.batch_size + 1) if num_samples > 0 \
        else (len(trn_loader.dataset) // trn_loader.batch_size)
    # Do forward and backward pass to accumulate L2-loss gradients
    model.train()
    for images, targets in itertools.islice(trn_loader, n_samples_batches):
        # MAS allows any unlabeled data to do the estimation, we choose the current data as in main experiments
        outputs = model.forward(images.to(device))
        # Page 6: labels not required, "...use the gradients of the squared L2-norm of the learned function output."
        loss = torch.norm(outputs, p=2, dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        # Eq. 2: accumulate the gradients over the inputs to obtain importance weights
        for n, p in model.named_parameters():
            if p.grad is not None:
                importance[n] += p.grad.abs() * len(targets)
    # Eq. 2: divide by N total number of samples
    n_samples = n_samples_batches * trn_loader.batch_size
    importance = {n: (p / n_samples) for n, p in importance.items()}
    return importance