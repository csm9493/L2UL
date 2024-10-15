# Learning to Unlearn: Instance-Wise Unlearning for Pre-trained Classifiers [AAAI 2024]

Sungmin Cha, Sungjun Cho, Dasol Hwang, Honglak Lee, Taesup Moon, and Moontae Lee

Paper: [[Arxiv](https://arxiv.org/abs/2301.11578)]

Abstract: Since the recent advent of regulations for data protection (e.g., the General Data Protection Regulation), there has been increasing demand in deleting information learned from sensitive data in pre-trained models without retraining from scratch. The inherent vulnerability of neural networks towards adversarial attacks and unfairness also calls for a robust method to remove or correct information in an instance-wise fashion, while retaining the predictive performance across remaining data. To this end, we consider instance-wise unlearning, of which the goal is to delete information on a set of instances from a pre-trained model, by either misclassifying each instance away from its original prediction or relabeling the instance to a different label. We also propose two methods that reduce forgetting on the remaining data: 1) utilizing adversarial examples to overcome forgetting at the representation-level and 2) leveraging weight importance metrics to pinpoint network parameters guilty of propagating unwanted information. Both methods only require the pre-trained model and data instances to forget, allowing painless application to real-life settings where the entire training set is unavailable. Through extensive experimentation on various image classification benchmarks, we show that our approach effectively preserves knowledge of remaining data while unlearning given instances in both single-task and continual unlearning scenarios.

-------


## Environment

See environment.yml

-------

## Implementation Guide

### Setting Up the Environment and Folder
1. Create the environment: conda env create -f environment.yml
2. Install necessary libraries: pip install transformers
3. Clone the repository: git clone https://github.com/csm9493/L2UL.git
4. Navigate to the project directory: cd L2UL
5. Create a folder for storing results: mkdir result_data

### Unlearning for ResNet-18/50 and ViT Models Trained on CIFAR-10/100 or ImageNet-1k
1. Navigate to the project directory: cd L2UL
2. Run the unlearning script: ./run_unlearn_cifar_imagenet.sh


### Unlearning for ResNet-18 Model Trained on the UTKFace Dataset
1. Navigate to the project directory: cd L2UL
2. Create a folder for dataset storage: mkdir data
3. Download the UTKFace dataset from [this link](https://www.kaggle.com/datasets/jangedoo/utkface-new) and place it in the './data/' directory.
4. Run the unlearning script: ./run_unlearn_utkface.sh

### *Reproducibility Issues Due to Experimental Setup*

Depending on the experimental environment, using the reported hyperparameter values for the L2UL algorithm may yield lower performance than what is presented in the paper. In such cases, we recommend to find new hyperparameter values by adjusting 'pgd_eps' to a range of 1.0 to 5.0 or 'unlearn_lr' to a range of 0.001 to 0.0001.


### Additional: Pretraining ResNet-18/50 Models on CIFAR-10/100 or UTKFace
#### Relevant scripts:

main_pretrain_cifar100_resnet50.py

main_pretrain_cifar10_resnet18.py

main_pretrain_utkface_resnet18.py

#### Note: For ViT models trained on CIFAR-100, we used the pre-trained model from HuggingFace(Ahmed9275/Vit-Cifar100).

-------
## Citation
@inproceedings{cha2024learning,
  title={Learning to unlearn: Instance-wise unlearning for pre-trained classifiers},
  author={Cha, Sungmin and Cho, Sungjun and Hwang, Dasol and Lee, Honglak and Moon, Taesup and Lee, Moontae},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={10},
  pages={11186--11194},
  year={2024}
}

-------