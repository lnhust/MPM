import click
import os, sys, glob
import numpy as np
import random
from collections import OrderedDict
import inspect
import pdb
import time
import re

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data as data

from dataloader import TaskSampler

from embedding_net import ConvNet
from inner_loop import InnerLoop

from score import *

class MetaLearner(object):
    def __init__(self,
                num_way,
                num_shot,
                classifier_param,
                loss_fn,
                inner_step_size,
                num_inner_updates,
                has_bias=False,
                alpha=[5,1,1],
                num_layer=1):
        super(self.__class__, self).__init__()
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_inner_updates = num_inner_updates
        self.inner_step_size = inner_step_size
        self.alpha = alpha
        
        # Make the nets
        #TODO: don't actually need two nets
        net_param_opt = {'userelu':False, 
            'in_planes':3, 'out_planes':[64,64,64,64], 'num_stages':4, 
            'num_dim':64*5*5, 'num_Kall':64
        }
        num_dim = net_param_opt['num_dim']

        self.model = {}
        self.model['embedding_net'] = ConvNet(net_param_opt)
        self.model['embedding_net'].cuda()

        self.model['classifier_net'] = classifier_param

        self.model['test_net'] = InnerLoop(num_dim, 5, loss_fn, self.num_inner_updates, 
            self.inner_step_size, 1, classifier_param, has_bias, self.alpha, num_layer)
        self.model['test_net'].cuda()


    def load_network(self, encoder_path):
        pretrained_embedding_net = torch.load(encoder_path)['network']
        self.model['embedding_net'].load_state_dict(pretrained_embedding_net)
        print('Load pretrained embedding net from {}.'.format(encoder_path))


    def test(self, dataset, exp_dir):
        self.model['embedding_net'].eval()

        mval_loss, mval_acc_novel = 0.0, 0.0
        num = len(os.listdir(exp_dir)) / 2
        for idx in range(num):
            print(idx)

            state = torch.load('{}/task-{}.pkl'.format(exp_dir, idx))
            self.model['test_net'].load_state_dict(state)
            sampler = TaskSampler('{}/task-{}.txt'.format(exp_dir, idx))
            dataloader = data.DataLoader(dataset, batch_sampler=sampler, shuffle=False)

            train_images, query_images, query_labels = [], [], []
            for i, batch in enumerate(dataloader):
                num_query = batch[0].shape[0] - self.num_shot
                query_images.append(batch[0][:num_query])
                train_images.append(batch[0][-self.num_shot:])
                query_labels.append(torch.LongTensor(np.array([i for _ in range(num_query)])))
            train_images = torch.stack(train_images).cuda()
            query_images = torch.cat(query_images).cuda()
            query_labels = torch.cat(query_labels).cuda()

            num_way,num_shot = train_images.size()[:2]
            with torch.no_grad():
                train_data = self.model['embedding_net'](
                    train_images.view((num_way*num_shot,)+train_images.size()[-3:]))
                train_data = train_data.view(num_way, num_shot, -1)
                query_data = self.model['embedding_net'](query_images)

                train_images_flip = train_images.flip([4])
                train_data_flip = self.model['embedding_net'](
                    train_images_flip.view(
                        (num_way*num_shot,)+train_images_flip.size()[-3:]
                    )
                )
                train_data_flip = train_data_flip.view(num_way, num_shot, -1)
                train_data = torch.cat([train_data, train_data_flip], dim=1)

            vloss, vacc = self.model['test_net'].evaluate(train_data, query_data, query_labels)


            mval_loss += vloss
            mval_acc_novel += vacc

        mval_loss = mval_loss / (idx+1)
        mval_acc_novel = mval_acc_novel / (idx+1)

        print('-------------------------')
        print(('%s. Validation. ' 
            'Val Loss %.5f. Val Acc %.4f.') % 
            (time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())), 
            mval_loss, mval_acc_novel))
        print('-------------------------')
        return mval_loss, mval_acc_novel
