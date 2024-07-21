from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

from advertorch.attacks import L2PGDAttack

from typing import *

from scipy.io import savemat
import copy

import itertools
from itertools import cycle
from resnet import resnet18, resnet34, resnet50

import numpy as np
import random

from utils import JointDataset, NormalizeLayer, naive_train, train, adv_attack, test, estimate_parameter_importance


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
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
    parser.add_argument('--unlearn-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--num-adv-images', type=int, default=None, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--reg-lamb', type=float, default=10.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
   
    eps = args.pgd_eps
    iters = args.pgd_iter
    alpha = args.pgd_alpha
    
    k_arr = [16]
#     k_arr = [1, 16, 64, 128, 256]
    
    D_r_acc = []
    D_f_acc = []
    D_test_acc = []
    
    case1_D_r = []
    case2_D_r = []
    case3_D_r = []
    
    case1_D_f = []
    case2_D_f = []
    case3_D_f = []
    
    case1_D_test = []
    case2_D_test = []
    case3_D_test = []
    

    train_kwargs = {'batch_size': 256}
    test_kwargs = {'batch_size': 1024}

    naiive_unlearn_kwargs = {'batch_size': 32}
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])

    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform)
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    for unlearn_k in k_arr:
        
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
        
        
        unlearn_label = args.unlearn_label

        train_labels = dataset1.targets
        
        train_labels = torch.from_numpy(np.array(train_labels))

        indices_k_unlearn = torch.randperm(train_labels.shape[0])[:unlearn_k]
        print ('indices_k_unlearn : ', indices_k_unlearn)

        copy_train_labels = train_labels.clone()
        copy_train_labels[indices_k_unlearn] = -10

        indices_other_data = (copy_train_labels != -10).nonzero(as_tuple=False)


        unlearn_dataset = Subset(dataset1, indices_k_unlearn.view(-1,))
        unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset,**naiive_unlearn_kwargs)

        other_dataset = Subset(dataset1, indices_other_data.view(-1,))
        other_loader = torch.utils.data.DataLoader(other_dataset,**test_kwargs)
        
        cifar_test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        
        print ('len(unlearn_dataset) : ', len(unlearn_dataset), ' len(other_dataset) : ', len(other_dataset))

        model = resnet18().to(device)
        model.load_state_dict(torch.load('./cifar10_pretrained_models/resnet18.pt'))
        normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        model = torch.nn.Sequential(normalize_layer, model)
        
        optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-4)
        
        model.eval()
        print ('Baseline 1: Naiive Appraoch - finetuning with D_forget (maximizing CE loss)')
            
        other_loss, other_acc = test(model, device, other_loader)
        unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)
        
        str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) +  ', D_forget acc : ' + str(unlearn_acc)+  ', D_test acc : ' +  str(test_acc)
        
        D_test_acc.append(test_acc)
        D_r_acc.append(other_acc)
        D_f_acc.append(unlearn_acc)
        
        print (str_list)

        unlearn_acc = 100
        max_iter = 1000
        
        j = 0

        while unlearn_acc != 0:

            naive_train(args, model, device, unlearn_loader, optimizer, 0)
            model.eval()
            unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
            
            j += 1
            
            if max_iter < j:
                
                break
                
        model.eval()
        other_loss, other_acc = test(model, device, other_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)

        str_list = '\n After | D_test - D_forget acc : ' + str(other_acc) +  ', D_forget acc : ' +  str(unlearn_acc) +  ', D_test acc : ' +  str(test_acc)
        print (str_list)
            
        case1_D_test.append(test_acc)
        case1_D_r.append(other_acc)
        case1_D_f.append(unlearn_acc)

        model = resnet18().to(device)
        model.load_state_dict(torch.load('./cifar10_pretrained_models/resnet18.pt'))
        normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        model = torch.nn.Sequential(normalize_layer, model)

        optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-4)
        
        origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        print ()
        print ('\n Baseline 2: Our Appraoch - using adversarial examples only')

        unlearn_acc = 100
        alpha = 0.0

        model.eval()
        other_loss, other_acc = test(model, device, other_loader)
        unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)
        
        str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) +  ', D_forget acc : ' +  str(unlearn_acc)+  ', D_test acc : ' +  str(test_acc)
        print (str_list)

        adversary = L2PGDAttack(model, eps=args.pgd_eps, eps_iter=args.pgd_alpha, nb_iter=args.pgd_iter,
                                rand_init=True, targeted=True)
        
        adv_images, target_labels = adv_attack(args, model, device, unlearn_loader, adversary, unlearn_k, args.num_adv_images)

        adv_dataset = JointDataset(adv_images, target_labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)

        j = 0
        
        unlearn_loader_cycle = cycle(unlearn_loader)
        CE = nn.CrossEntropyLoss()
        
        while unlearn_acc != 0:
            model.train()

            for i , data in enumerate(zip(adv_loader, unlearn_loader_cycle)):
                
                model.train()
                
                
                (adv_data, adv_target), (data, target) = data

                optimizer.zero_grad()

                output_adv = model(adv_data.to(device))
                output = model(data.to(device))

                loss_unlearn = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
                loss_adv = CE(output_adv, adv_target.to(device)) * (adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))

                loss = loss_unlearn + loss_adv

                loss.backward()
                optimizer.step()
                
                model.eval()
                unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
                
                if unlearn_acc == 0:
                    print ('unlearn_acc == 0, Break at j = ', j, ' i = ', i)
                    break

            
            j += 1
            
            if max_iter < j:
                
                break

        model.eval()
        unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
        other_loss, other_acc = test(model, device, other_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)
        str_list = '\n After | D_test - D_forget acc : ' + str(other_acc) +  ', D_forget acc : ' +  str(unlearn_acc)+  ', D_test acc : ' +  str(test_acc)
        print (str_list)
            
        
        case2_D_test.append(test_acc)
        case2_D_r.append(other_acc)
        case2_D_f.append(unlearn_acc)

        print ()
        print ('\n Baseline 3: Our Appraoch - using both adversarial examples and weight importance')

        model = resnet18().to(device)
        model.load_state_dict(torch.load('./cifar10_pretrained_models/resnet18.pt'))
        normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        model = torch.nn.Sequential(normalize_layer, model)

        optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-4)
        
        origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        model_for_importance = copy.deepcopy(model)
        num_samples = len(unlearn_dataset)
        importance = estimate_parameter_importance(unlearn_loader, model_for_importance, device, num_samples, optimizer)
        

        for keys in importance.keys():
            
            importance[keys] = (importance[keys] - importance[keys].min()) / (importance[keys].max() - importance[keys].min())
            importance[keys] = (1 - importance[keys])
                    
        CE = nn.CrossEntropyLoss()

        unlearn_acc = 100

        model.eval()
        other_loss, other_acc = test(model, device, other_loader)
        unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)
        
        str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) +  ', D_forget acc : ' +  str(unlearn_acc)+  ', D_test acc : ' +  str(test_acc)
        print (str_list)

        adv_dataset = JointDataset(adv_images, target_labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)

        j = 0
        unlearn_loader_cycle = cycle(unlearn_loader)
        
        while unlearn_acc != 0:

            for i , data in enumerate(zip(adv_loader, unlearn_loader_cycle)):
                
                model.train()
                
                (adv_data, adv_target), (data, target) = data

                optimizer.zero_grad()

                output_adv = model(adv_data.to(device))
                output = model(data.to(device))

                loss_unlearn = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
                loss_adv = CE(output_adv, adv_target.to(device)) * (adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))


                loss_reg = 0

                for n, p in model.named_parameters():
                    if n in importance.keys():
                        loss_reg += torch.sum(importance[n] * (p - origin_params[n]).pow(2)) / 2

                loss = loss_unlearn + loss_adv + loss_reg * args.reg_lamb

                loss.backward()
                optimizer.step()
                
                
                model.eval()
                unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
                
                if unlearn_acc == 0:
                    print ('unlearn_acc == 0, Break at j = ', j, ' i = ', i)
                    break

            
            j += 1
            
            if max_iter < j:
                
                break

        model.eval()
        unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
        other_loss, other_acc = test(model, device, other_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)

        str_list = '\n After | D_test - D_forget acc : ' + str(other_acc) +  ', D_forget acc : ' +  str(unlearn_acc)+  ', D_test acc : ' +  str(test_acc)
        print (str_list)
            
        
        case3_D_test.append(test_acc)
        case3_D_r.append(other_acc)
        case3_D_f.append(unlearn_acc)
        
        save_file_name = 'cifar10_unlearning_label_mix_unlearning_lr_' + str(args.unlearn_lr) + '_reg_lamb_' + str(args.reg_lamb) + '_num_adv_images_' + str(args.num_adv_images) + '_l2_pgd_eps_' + str(args.pgd_eps) + '_iter_' + str(args.pgd_iter) + '_alpha_' + str(args.pgd_alpha)  + '_seed_' + str(args.seed)
        
        savemat('./result_data/' + save_file_name + '.mat', {"k_arr": np.array(k_arr),"D_r_acc": np.array(D_r_acc),"D_test_acc": np.array(D_test_acc),"D_f_acc": np.array(D_f_acc),"case1_D_r": np.array(case1_D_r),"case2_D_r": np.array(case2_D_r),"case3_D_r": np.array(case3_D_r),"case1_D_f": np.array(case1_D_f),"case2_D_f": np.array(case2_D_f),"case3_D_f": np.array(case3_D_f),"case1_D_test": np.array(case1_D_test),"case2_D_test": np.array(case2_D_test),"case3_D_test": np.array(case3_D_test)})


if __name__ == '__main__':
    main()