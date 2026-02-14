# Distributed SGD with PyTorch DDP – ML Systems Optimization Assignment

Implementation of synchronous data-parallel SGD using PyTorch DistributedDataParallel (DDP).

- Dataset: CIFAR-10 (reduced to ~5,000 images for fast demo in Colab)
- Model: ResNet-18
- Measured training time for 1 epoch (single-process baseline)

## How to run
- Open in Colab:  
  https://colab.research.google.com/github/2024ac05341/distributed-sgd-ddp-cifar10/blob/main/ddp_cifar_resnet.ipynb
- Set runtime to T4 GPU
- Run all cells
- See printed training time

Full report PDF attached in submission.
