import numpy as np
import os
import random
import torch
import torch.nn as nn
import argparse
import pdb

from mpm import MetaLearner
from dataloader import MiniImageNet,TieredImageNet,FewShotDataloader

def get_dataset(split='train', dataset='miniImageNet'):
    if dataset == 'miniImageNet':
        dataset = MiniImageNet(phase=split)
    elif dataset == 'tieredImageNet':
        dataset = TieredImageNet(phase=split)
    else:
        print 'Unknown dataset'
        raise(Exception)

    return dataset

def main(args):

    random.seed(1337)
    np.random.seed(1337)

    print 'Setting GPU to', str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    learner = torch.load(args.model_path)
    print('Load model from {}.'.format(args.model_path))

    test_dataset = get_dataset('test', args.dataset)
    learner.test(test_dataset, args.exp_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='output/ConvNet/miniImageNet_test_1shot.model',
        help='output directory to save trained models')
    parser.add_argument('--num-cls', type=int, default=5, help='N way')
    parser.add_argument('--num-inst', type=int, default=1, help='K shot')
    parser.add_argument('--num-query', type=int, default=15,
        help='the number of queries per class')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
        help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--exp-dir', type=str, default='output/ConvNet/miniImageNet-1shot', help='gpu device')

    args = parser.parse_args()
    main(args)
