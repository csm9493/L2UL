from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader, Dataset, Subset, random_split

from typing import *

from scipy.io import savemat
import copy
import glob

import itertools
from itertools import cycle
from resnet_cifar100 import resnet18, resnet34, resnet50

import numpy as np
import random
import wandb

import pandas as pd
from sklearn.model_selection import train_test_split

from scipy.io import savemat

from tqdm import tqdm
from PIL import Image

from utils import (JointDataset,
                   NormalizeLayer,
                   adv_attack,
                   FeatureExtractor,
                   ViTModel,
                   estimate_parameter_importance,
                   getDataLoaders,
                   UTKDataset)

def run_model(model, data_loader, device, optimizer=None, lr_scheduler=None, grad_clip=None):
    
    total_loss = 0
    total_correct = 0
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        data, target = batch[0].to(device), batch[1].to(device)
        bsz = data.shape[0]
            
        output = model(data)

        loss = nn.CrossEntropyLoss()(output, target)

        if not optimizer is None:
            loss.backward()

            if grad_clip: 
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        num_correct = pred.eq(target.view_as(pred)).sum().item()

        total_loss += loss.item() * bsz
        total_correct += num_correct

    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_correct / len(data_loader.dataset)
    
    return avg_loss, avg_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=1e-3, metavar='WD',
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--grad-clip', type=float, default=0.1, metavar='GC',
                        help='gradient clipping (default:0.1)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                        help='interval for evaluation (default: 1)')
    #parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                    help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--age-grouping', type=str, default='MFD',
                        help='which age grouping to use for age classification')
    
    parser.add_argument('--project', type=str, default='unlearning',
                        help='project name for wandb logging')
    parser.add_argument('--exp-name', type=str, default='resnet-18-utkface',
                        help='run name for wandb logging')
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    device_ids = list(range(torch.cuda.device_count()))
    print(f"GPU list: {device_ids}")

    ckpt_filename = f"checkpoints/resnet-18_utkface_group-{args.age_grouping}_lr-{args.lr}_wd-{args.wd}.pt"
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    # List of Transformations (Augmentation)
    train_transform = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.RandomCrop((120, 120)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Lambda(lambda x: x/255.)])
    test_transform = transforms.Compose([transforms.Resize((128, 128)),
                                         transforms.CenterCrop((120, 120)),
                                         transforms.ToTensor(),
                                         transforms.Lambda(lambda x: x/255.)])

    # Load datasets and split
    train_dataset = UTKDataset('./data/UTKFace', args.age_grouping, train_transform)
    valid_dataset = UTKDataset('./data/UTKFace', args.age_grouping, test_transform)
    test_dataset = UTKDataset('./data/UTKFace', args.age_grouping, test_transform)
    NUM_CLASSES = len(train_dataset.bins) - 1

    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx = indices[:18966], indices[18966:21337], indices[21337:]
    train_dataset = Subset(train_dataset, indices=train_idx)
    valid_dataset = Subset(valid_dataset, indices=valid_idx)
    test_dataset = Subset(test_dataset, indices=test_idx)
    """
    train_data, valid_data, test_data = random_split(dataset,
                                                     [18966, 2371, 2371],
                                                     generator=torch.Generator().manual_seed(42))
    """

    if use_cuda:
        cuda_kwargs = {'num_workers': 32,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Load the datasets into dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, **train_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **test_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **test_kwargs)

    wandb.init(project=args.project)
    wandb.run.name = args.exp_name
    wandb.config.update(args)
        
    # INITIALIZE MODEL
    model = resnet18(NUM_CLASSES)
    """
    ckpt = torch.load('./pretrained_models/resnet18.pt')
    fc_names = [key for key in ckpt.keys() if "fc" in key]
    for key in fc_names:
        del ckpt[key]
    model.load_state_dict(ckpt, strict=False)
    normalize_layer = NormalizeLayer((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    model = torch.nn.Sequential(normalize_layer, model)
    """
    model = model.cuda()

    # INITIALIZE OPTIMIZER + LR_SCHEDULER
    """
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=0.9, 
                          weight_decay=args.wd)
    """

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                 args.lr, 
                                                 epochs=args.epochs,
                                                 steps_per_epoch=len(train_loader))
    
    
    """
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    """

    best_val_acc = 0
    for epoch in range(args.epochs):

        model.train()
        train_loss, train_acc = run_model(model, 
                                          train_loader, 
                                          device, 
                                          optimizer=optimizer,
                                          lr_scheduler=lr_scheduler,
                                          grad_clip=args.grad_clip)
        print(f"Epoch {epoch+1}: Train loss = {train_loss} / Train Acc. = {train_acc}")
        wandb.log({
            'epoch': epoch+1,
            'learning_rate': lr_scheduler.get_last_lr()[0],
            'train_loss': train_loss,
            'train_acc': train_acc,
        })

        ### Run on validation step
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            val_loss, val_acc = run_model(model, valid_loader, device)
            print(f"Valid loss = {val_loss} / Valid Acc. = {val_acc}")
            wandb.log({
                'epoch': epoch+1,
                'val_loss': val_loss,
                'val_acc': val_acc,
            })

            if val_acc > best_val_acc:
                print(f"Found new best validation score! Saving checkpoint to {ckpt_filename}...")
                torch.save(model.state_dict(), ckpt_filename)
                best_val_loss = val_loss
                best_val_acc = val_acc

    print(f"Optimization Finished! Loading checkpoint from {ckpt_filename} with best validation loss...")
    model.load_state_dict(torch.load(ckpt_filename))
    model.eval()
    test_loss, test_acc = run_model(model, test_loader, device)
    print(f"Test loss = {test_loss} / Test Acc. = {test_acc}\n")
    wandb.log({
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
    })


if __name__ == '__main__':
    main()