# Learning to Unlearn: Instance-Wise Unlearning for Pre-trained Classifiers 

Sungmin Cha, Sungjun Cho, Dasol Hwang, Honglak Lee and Taesup Moon, Moontae Lee
Paper: [[Arxiv](https://arxiv.org/abs/2301.11578)]

Abstract: Since the recent advent of regulations for data protection (e.g., the General Data Protection Regulation), there has been increasing demand in deleting information learned from sensitive data in pre-trained models without retraining from scratch. The inherent vulnerability of neural networks towards adversarial attacks and unfairness also calls for a robust method to remove or correct information in an instance-wise fashion, while retaining the predictive performance across remaining data. To this end, we consider instance-wise unlearning, of which the goal is to delete information on a set of instances from a pre-trained model, by either misclassifying each instance away from its original prediction or relabeling the instance to a different label. We also propose two methods that reduce forgetting on the remaining data: 1) utilizing adversarial examples to overcome forgetting at the representation-level and 2) leveraging weight importance metrics to pinpoint network parameters guilty of propagating unwanted information. Both methods only require the pre-trained model and data instances to forget, allowing painless application to real-life settings where the entire training set is unavailable. Through extensive experimentation on various image classification benchmarks, we show that our approach effectively preserves knowledge of remaining data while unlearning given instances in both single-task and continual unlearning scenarios.

-------


## Environment

See environment.yml

-------

## How to implement?

1. conda env create -f environment.yml
2. git clone https://github.com/csm9493/L2UL.git
3. cd L2UL
4. mkdir result_data
5. ./run.sh

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