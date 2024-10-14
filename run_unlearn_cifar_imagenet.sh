
# CIFAR-10
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar10_mixed_label_resnet18.py  --num-adv-images 20  --pgd-eps 4.0  --unlearn-lr 0.001 --reg-lamb 1.0 --seed 0

# CIFAR-100
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar100_mixed_label_resnet50.py --num-adv-images 200  --pgd-eps 4.0  --unlearn-lr 0.001 --reg-lamb 1.0 --seed 0 
CUDA_VISIBLE_DEVICES=0 python main_unlearn_cifar100_mixed_label_vit.py --num-adv-images 200  --pgd-eps 4.0  --unlearn-lr 0.001 --reg-lamb 1.0 --seed 0 

# ImageNet-1k
CUDA_VISIBLE_DEVICES=0 python main_unlearn_imagenet_mixed_label_resnet50.py --num-adv-images 200  --pgd-eps 4.0  --unlearn-lr 0.001 --reg-lamb 1.0 --seed 0