
# Train ResNet-18 using UTKFace dataset
CUDA_VISIBLE_DEVICES=0 python main_pretrain_utkface_resnet18.py --age_grouping 'TNN' --lr 0.005 --wd 0.0001 --seed 0
CUDA_VISIBLE_DEVICES=0 python main_pretrain_utkface_resnet18.py --age_grouping 'MFD' --lr 0.0001 --wd 0.0 --seed 0
CUDA_VISIBLE_DEVICES=0 python main_pretrain_utkface_resnet18.py --age_grouping 'balanced' --lr 0.001 --wd 0.0 --seed 0
CUDA_VISIBLE_DEVICES=0 python main_pretrain_utkface_resnet18.py --age_grouping 'groups' --lr 0.001 --wd 0.0001 --seed 0
CUDA_VISIBLE_DEVICES=0 python main_pretrain_utkface_resnet18.py --age_grouping 'tens' --lr 0.001 --wd 0.001 --seed 0


# Unlearning
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar10_mixed_label_resnet18.py  --age_grouping 'TNN' --seed 0 --num-adv-images 300  --pgd-eps 2.0  --unlearn-lr 0.001 --reg-lamb 1.0
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar10_mixed_label_resnet18.py  --age_grouping 'MFD' --seed 0 --num-adv-images 300  --pgd-eps 2.0  --unlearn-lr 0.001 --reg-lamb 1.0
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar10_mixed_label_resnet18.py  --age_grouping 'balanced' --seed 0 --num-adv-images 300  --pgd-eps 2.0  --unlearn-lr 0.001 --reg-lamb 1.0
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar10_mixed_label_resnet18.py  --age_grouping 'groups' --seed 0 --num-adv-images 300  --pgd-eps 2.0  --unlearn-lr 0.001 --reg-lamb 1.0
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar10_mixed_label_resnet18.py  --age_grouping 'tens' --seed 0 --num-adv-images 300  --pgd-eps 2.0  --unlearn-lr 0.001 --reg-lamb 1.0