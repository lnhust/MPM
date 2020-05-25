import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import *
from score import *
import pdb

MAX = 1000
END2END = False
class InnerLoop(nn.Module):
    '''
    This module performs the inner loop of MAML
    The forward method updates weights with gradient steps on training data, 
    then computes and returns a meta-gradient w.r.t. validation data
    '''

    def __init__(self, n_dim, n_way, loss_fn, num_updates, step_size, meta_batch_size, classifier=None, bias=True, alpha=[5,1,1], n_layer=1):
        super(InnerLoop, self).__init__()
        self.n_dim = n_dim
        self.n_way = n_way
        self.has_bias = bias
        self.alpha = alpha

        if n_layer == 1:
            self.n_layer = 1
            self.layers = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.n_dim, self.n_dim, bias=bias))]))
                for i in range(self.n_way)
            ])
        else:
            self.n_layer = 2
            self.layers = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.n_dim, self.n_dim, bias=bias)),
                    ('relu', nn.ReLU()),
                    ('fc2', nn.Linear(self.n_dim, self.n_dim, bias=bias))]))
                for i in range(self.n_way)
            ])
            self.relu = nn.ReLU()

        if classifier is None:
            weight_base = torch.FloatTensor(MAX, n_dim).normal_(0.0, np.sqrt(2.0/n_dim))
            scale_cls = torch.FloatTensor(1).fill_(10)
            bias = torch.FloatTensor(1).fill_(0)
        else:
            weight_base = torch.FloatTensor(classifier['weight_base'])
            scale_cls = torch.FloatTensor(1).fill_(classifier['scale_cls'])
            bias = torch.FloatTensor(1).fill_(classifier['bias'])
        self.weight_base = nn.Parameter(weight_base, requires_grad=END2END)
        self.scale_cls = nn.Parameter(scale_cls, requires_grad=END2END)
        self.bias = nn.Parameter(bias, requires_grad=END2END)

        self.loss_fn = loss_fn

        # Number of updates to be taken
        self.num_updates = num_updates

        # Step size for the updates
        self.step_size = step_size

        # for loss normalization 
        self.meta_batch_size = meta_batch_size


    def net_forward(self, train_data):
        ''' Run train data through net, return transformed data 
             train_data: [n_way, n_shot, n_dim]
        '''
        train_data = F.normalize(train_data, dim=-1, p=2, eps=1e-8)

        train_data_transformed = []
        for i in range(self.n_way):
            piece = self.layers[i](train_data[i])
            train_data_transformed.append(piece)
        train_data_transformed = torch.stack(train_data_transformed)

        return train_data_transformed

    def net_forward_classify(self, train_data, query_data, query_label):
        ''' Run data through net, return loss and output for classification
             train_data: [n_way, n_shot, n_dim]
             query_data: [n_query, n_dim], query_label: [n_query]
        '''
        train_data_transformed = self.net_forward(train_data)
        weight_novel = F.normalize(train_data_transformed, dim=-1, p=2, eps=1e-8).mean(1)
        weight_novel = F.normalize(weight_novel, dim=-1, p=2, eps=1e-8)

        query_data = F.normalize(query_data, dim=-1, p=2, eps=1e-8)
        scores = torch.matmul(query_data, weight_novel.transpose(0,1))
        scores = self.scale_cls * (scores + self.bias)
        loss = self.loss_fn(scores, query_label)

        return loss,scores
    
    def evaluate(self, train_data, query_data, query_label):
        ''' evaluate the net on the data in the loader '''
        val_loss, scores = self.net_forward_classify(train_data, query_data, query_label)

        num_correct = count_correct(np.argmax(scores.data.cpu().numpy(), axis=1), query_label.cpu().data.numpy())
        val_acc = float(num_correct) / query_label.size(0)

        return val_loss.item(), val_acc
