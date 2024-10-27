from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from advertorch.attacks import L2PGDAttack
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

from typing import *

from scipy.io import savemat
import copy

import itertools
from itertools import cycle
from resnet_cifar100 import resnet18, resnet34, resnet50

import numpy as np
import random


from scipy.io import savemat


from utils import (JointDataset,
                   NormalizeLayer,
                   naive_train,
                   train,
                   test,
                   adv_attack,
                   FeatureExtractor,
                   ViTModel,
                   estimate_parameter_importance,
                   getDataLoaders,
                   UTKDataset)

def getResNet18(lr, num_classes, age_grouping):
    model = resnet18(num_classes)
    if age_grouping == 'TNN':
        ckpt = f'./checkpoints/resnet-18_utkface_group-TNN_lr-0.005_wd-0.0001.pt'
    elif age_grouping == 'MFD':
        ckpt = f'./checkpoints/resnet-18_utkface_group-MFD_lr-0.0001_wd-0.0.pt'
    elif age_grouping == 'balanced':
        ckpt = f'./checkpoints/resnet-18_utkface_group-balanced_lr-0.001_wd-0.0.pt'
    elif age_grouping == 'groups':
        ckpt = f'./checkpoints/resnet-18_utkface_group-groups_lr-0.001_wd-0.0001.pt'
    elif age_grouping == 'tens':
        ckpt = f'./checkpoints/resnet-18_utkface_group-tens_lr-0.001_wd-0.001.pt'
    else:
        raise NotImplementedError
        
    model.load_state_dict(torch.load(ckpt))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    return model, optimizer

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--age-grouping', type=str, default='MFD',
                        help='which age grouping to use for age classification')
    
    parser.add_argument('--pgd-eps', type=float, default=2.0, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--pgd-alpha', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--pgd-iter', type=int, default=100, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    
    parser.add_argument('--unlearn-label', type=int, default=9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--unlearn-k', type=int, default=10, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--unlearn-lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--num-adv-images', type=int, default=300, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--reg-lamb', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    device_ids = list(range(torch.cuda.device_count()))
    print(f"GPU list: {device_ids}")
   
    eps = args.pgd_eps
    iters = args.pgd_iter
    alpha = args.pgd_alpha
    
    # k: number of unlearning data
    k_arr = [16]
#     k_arr = [16, 32, 64, 128, 256]

    Dr_acc = []
    Df_acc = []
    Dt_acc = []
    case0_Dr = []
    case0_Df = []
    case0_Dt = []
    case1_Dr = []
    case1_Df = []
    case1_Dt = []
    case2_Dr = []
    case2_Df = []
    case2_Dt = []
    case3_Dr = []
    case3_Df = []
    case3_Dt = []

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    naive_unlearn_kwargs = {'batch_size': args.batch_size}
    
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

    train_dataset = UTKDataset('./data/UTKFace', args.age_grouping, train_transform)
    valid_dataset = UTKDataset('./data/UTKFace', args.age_grouping, test_transform)
    test_dataset = UTKDataset('./data/UTKFace', args.age_grouping, test_transform)
    NUM_CLASSES = len(train_dataset.bins) - 1

    print(f'Loaded UTKFace Dataset using {args.age_grouping} with {NUM_CLASSES} classes')
    print(f'PGD-eps = ', args.pgd_eps, f' / Unlearning LR = ', args.unlearn_lr)

    indices = list(range(len(train_dataset)))
    np.random.seed(0)
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx = indices[:18966], indices[18966:21337], indices[21337:]

    train_dataset = Subset(train_dataset, indices=train_idx)
    #valid_dataset = Subset(valid_dataset, indices=valid_idx)
    test_dataset = Subset(test_dataset, indices=test_idx)
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        naive_unlearn_kwargs.update(cuda_kwargs)


    ### Gather adversarial images
    print('Generating adversarial images...')
    np.random.seed(args.seed)
    rand_indices = torch.randperm(len(train_dataset))
    f_indices = rand_indices[:max(k_arr)]
    f_dataset = Subset(train_dataset, indices=f_indices)
    f_loader = DataLoader(f_dataset, shuffle=False, **train_kwargs)

    model, optimizer = getResNet18(args.unlearn_lr, NUM_CLASSES, args.age_grouping)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()
        
    # Set adversarial attack (L2 PGD Attack)
    adversary = L2PGDAttack(model, 
                            eps=args.pgd_eps, 
                            eps_iter=args.pgd_alpha, 
                            nb_iter=args.pgd_iter,
                            rand_init=True, 
                            targeted=True)
    
    adv_images_max, target_labels_max, image_idx_max = adv_attack(args, model, device, 
                                                                  f_loader, adversary, max(k_arr),
                                                                  num_classes=NUM_CLASSES,
                                                                  num_adv_images=args.num_adv_images,
                                                                  indices=f_indices)
    # len(adv_images) = len(target_labels) = unlearn_k * num_adv_images
    #assert set(f_indices) == set(image_idx_max)
    
    
    for unlearn_k in k_arr:

        # Load Datasets
        #rand_indices = torch.randperm(len(train_dataset))
        f_indices = rand_indices[:unlearn_k]
        r_indices = rand_indices[unlearn_k:]

        f_dataset = Subset(train_dataset, indices=f_indices)
        r_dataset = Subset(train_dataset, indices=r_indices)

        f_loader = DataLoader(f_dataset, **train_kwargs)
        r_loader = DataLoader(r_dataset, **test_kwargs)
        t_loader = DataLoader(test_dataset, **test_kwargs)

        print ('len(forget_dataset) : ', len(f_dataset), ' ',
               'len(residual_dataset) : ', len(r_dataset), ' ',
               'len(test_dataset) : ', len(test_dataset))

        # Filter out images based on their indices in f_indices
        adv_images = []
        target_labels = []
        for i in range(len(adv_images_max)):
            if image_idx_max[i] in f_indices:
                adv_images.append(adv_images_max[i])
                target_labels.append(target_labels_max[i])
        print('len(adv_images) : ', len(adv_images), ' ',
              'len(target_labels) : ', len(target_labels))
        
        max_iter = 1000
        f_loader_cycle = cycle(f_loader)
        CE = nn.CrossEntropyLoss()
        
        ############################# Before Unlearning #############################
        print('\nBefore Unlearning')
        # LOAD MODEL
        model, optimizer = getResNet18(args.unlearn_lr, NUM_CLASSES, args.age_grouping)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()
        
        r_loss, r_acc = test(model, device, r_loader)
        f_loss, f_acc = test(model, device, f_loader)
        t_loss, t_acc = test(model, device, t_loader)
        
        print('Before unlearning...',
              '\n - D_residual acc : ', str(r_acc),
              '\n - D_forget acc : ', str(f_acc),
              '\n - D_test acc : ',  str(t_acc))
        
        Dt_acc.append(t_acc)
        Dr_acc.append(r_acc)
        Df_acc.append(f_acc)
        
        ################### Case 0: Oracle / Loss = CE(D_r)-CE(D_f) ###################
        print('\nCase 0: Oracle - finetuning with D_r and D_f')
                   
        # LOAD MODEL
        model, optimizer = getResNet18(args.unlearn_lr, NUM_CLASSES, args.age_grouping)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()

        f_acc = 100
        j = 0
        while f_acc != 0:
            model.train()
            for i, data in enumerate(zip(r_loader, f_loader_cycle)):
                model.train()
                (r_data, r_target), (data, target) = data
                optimizer.zero_grad()
                r_output = model(r_data.to(device))
                output = model(data.to(device))
                f_loss = -CE(output, target.to(device)) 
                r_loss = CE(r_output, r_target.to(device)) 
                loss = f_loss + r_loss
                loss.backward()
                optimizer.step()
                model.eval()
                f_loss, f_acc = test(model, device, f_loader)
                
                if f_acc == 0:
                    print ('f_acc == 0... Breaking at epoch ', j, ' / example ', i)
                    break
            j += 1
            if max_iter < j:
                break
                
        model.eval()
        r_loss, r_acc = test(model, device, r_loader)
        t_loss, t_acc = test(model, device, t_loader)

        print('After unlearning with Case 0...',
              '\n - D_residual acc : ', str(r_acc),
              '\n - D_forget acc : ', str(f_acc),
              '\n - D_test acc : ',  str(t_acc))
            
        case0_Dt.append(t_acc)
        case0_Dr.append(r_acc)
        case0_Df.append(f_acc)
        
        #################### Case 1: Naive Approach / Loss = -CE(D_f) ####################
        print('\nCase 1: Naive Approach - finetuning with D_forget (maximizing CE loss)')
        # LOAD MODEL
        model, optimizer = getResNet18(args.unlearn_lr, NUM_CLASSES, args.age_grouping)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()
        
        f_acc = 100
        j = 0
        while f_acc != 0:
            model.train()
            naive_train(args, model, device, f_loader, optimizer, 0)
            model.eval()
            f_loss, f_acc = test(model, device, f_loader)
            j += 1
            
            if f_acc == 0:
                    print ('f_acc == 0... Breaking at epoch ', j)
                    break
            
            if max_iter < j:
                break
                
        model.eval()
        r_loss, r_acc = test(model, device, r_loader)
        t_loss, t_acc = test(model, device, t_loader)

        print('After unlearning with Case 1...',
              '\n - D_residual acc : ', str(r_acc),
              '\n - D_forget acc : ', str(f_acc),
              '\n - D_test acc : ',  str(t_acc))
            
        case1_Dt.append(t_acc)
        case1_Dr.append(r_acc)
        case1_Df.append(f_acc)
        
        ########### Case 2: with adversarial examples / Loss = -CE(D_f)+CE(D_adv) ###########
        print ('\nCase 2: Our Approach - using adversarial examples only')
        # LOAD MODEL
        model, optimizer = getResNet18(args.unlearn_lr, NUM_CLASSES, args.age_grouping)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()

        """
        # Set adversarial attack (L2 PGD Attack)
        adversary = L2PGDAttack(model, 
                                eps=args.pgd_eps, 
                                eps_iter=args.pgd_alpha, 
                                nb_iter=args.pgd_iter,
                                rand_init=True, 
                                targeted=True)
        
        adv_images, target_labels = adv_attack(args, 
                                               model, 
                                               device, 
                                               f_loader, 
                                               adversary, 
                                               unlearn_k,
                                               num_classes=NUM_CLASSES,
                                               num_adv_images=args.num_adv_images)
        """

        adv_dataset = JointDataset(adv_images, target_labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)
        
        f_acc = 100
        j = 0
        while f_acc != 0:
            model.train()
            for i , data in enumerate(zip(adv_loader, f_loader_cycle)):
                model.train()
                (adv_data, adv_target), (data, target) = data
                optimizer.zero_grad()
                output_adv = model(adv_data.to(device))
                output = model(data.to(device))

                loss_f = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
                loss_adv = CE(output_adv, adv_target.to(device)) * (adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))

                loss = loss_f + loss_adv
                loss.backward()
                optimizer.step()
                model.eval()
                f_loss, f_acc = test(model, device, f_loader)
                
                if f_acc == 0:
                    print ('f_acc == 0... Breaking at epoch ', j, ' / example ', i)
                    break
            j += 1
            if max_iter < j:
                break
                
        model.eval()
        r_loss, r_acc = test(model, device, r_loader)
        t_loss, t_acc = test(model, device, t_loader)

        print('After unlearning with Case 2...',
              '\n - D_residual acc : ', str(r_acc),
              '\n - D_forget acc : ', str(f_acc),
              '\n - D_test acc : ',  str(t_acc))
            
        case2_Dt.append(t_acc)
        case2_Dr.append(r_acc)
        case2_Df.append(f_acc)
        
        
        ########## Case 3: with adversarial examples + weight importance ##########
        ############## Loss = -CE(D_f) + CE(D_adv) + reg(importance) ##############
        print ('\nCase 3: Our Appraoch - using both adversarial examples and weight importance')

        # LOAD MODEL
        model, optimizer = getResNet18(args.unlearn_lr, NUM_CLASSES, args.age_grouping)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()
        
        #Set original parameters
        origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        #Get weight importance via MAS
        model_for_importance = copy.deepcopy(model)
        num_samples = len(f_loader.dataset)
        importance = estimate_parameter_importance(f_loader, model_for_importance, device, num_samples, optimizer)
        
        #Normalize weight importance by each layer and reverse it
        # Importance => 1 : not important to D_f / Importance => 0 : Important to D_f
        for keys in importance.keys():
            importance[keys] = (importance[keys] - importance[keys].min()) / (importance[keys].max() - importance[keys].min())
            importance[keys] = 1 - importance[keys]

        """
        # Set adversarial attack (L2 PGD Attack)
        adversary = L2PGDAttack(model, 
                                eps=args.pgd_eps, 
                                eps_iter=args.pgd_alpha, 
                                nb_iter=args.pgd_iter,
                                rand_init=True, 
                                targeted=True)
        
        adv_images, target_labels = adv_attack(args, 
                                               model, 
                                               device, 
                                               f_loader, 
                                               adversary, 
                                               unlearn_k, 
                                               args.num_adv_images)
        """

        adv_dataset = JointDataset(adv_images, target_labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)

        f_acc = 100
        j = 0        
        while f_acc != 0:
            for i , data in enumerate(zip(adv_loader, f_loader_cycle)):
                model.train()
                (adv_data, adv_target), (data, target) = data
                optimizer.zero_grad()
                output_adv = model(adv_data.to(device))
                output = model(data.to(device))

                loss_f = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
                loss_adv = CE(output_adv, adv_target.to(device)) * (adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))

                loss_reg = 0
                for n, p in model.named_parameters():
                    if n in importance.keys():
                        loss_reg += torch.sum(importance[n] * (p - origin_params[n]).pow(2)) / 2
                loss = loss_f + loss_adv + loss_reg * args.reg_lamb
                loss.backward()
                optimizer.step()
                model.eval()
                f_loss, f_acc = test(model, device, f_loader)
                
                if f_acc == 0:
                    print ('f_acc == 0... Breaking at epoch ', j, ' / example ', i)
                    break
            j += 1
            if max_iter < j:
                break

        model.eval()
        r_loss, r_acc = test(model, device, r_loader)
        t_loss, t_acc = test(model, device, t_loader)

        print('After unlearning with Case 3...',
              '\n - D_residual acc : ', str(r_acc),
              '\n - D_forget acc : ', str(f_acc),
              '\n - D_test acc : ',  str(t_acc))
            
        case3_Dt.append(t_acc)
        case3_Dr.append(r_acc)
        case3_Df.append(f_acc)
    
    print('k_arr', k_arr)
    print('Dr_acc', Dr_acc)
    print('Df_acc', Df_acc)
    print('Dt_acc', Dt_acc)
    print('case0_Dr', case0_Dr)
    print('case0_Df', case0_Df)
    print('case0_Dt', case0_Dt)
    print('case1_Dr', case1_Dr)
    print('case1_Df', case1_Df)
    print('case1_Dt', case1_Dt)
    print('case2_Dr', case2_Dr)
    print('case2_Df', case2_Df)
    print('case2_Dt', case2_Dt)
    print('case3_Dr', case3_Dr)
    print('case3_Df', case3_Df)
    print('case3_Dt', case3_Dt)


if __name__ == '__main__':
    main()