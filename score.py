import numpy as np

import torch
from torch.autograd import Variable

'''
Helper methods for evaluating a classification network
'''

def count_correct(pred, target):
    ''' count number of correct classification predictions in a batch '''
    pairs = [int(x==y) for (x, y) in zip(pred, target)]
    return sum(pairs)

def forward_pass(net, in_, target, weights=None):
    ''' forward in_ through the net, return loss and output '''
    input_var = Variable(in_).cuda(async=True)
    target_var = Variable(target).cuda(async=True)
    out = net.net_forward(input_var, weights)
    loss = net.loss_fn(out, target_var)
    return loss, out

def evaluate(net, loader, weights=None):
    ''' evaluate the net on the data in the loader '''
    num_correct = 0
    loss = 0
    for i, (in_, target) in enumerate(loader):
        batch_size = in_.numpy().shape[0]
        l, out = forward_pass(net, in_, target, weights)
        loss += l.data.cpu().numpy()[0]
        num_correct += count_correct(np.argmax(out.data.cpu().numpy(), axis=1), target.numpy())
    return float(loss) / len(loader), float(num_correct) / (len(loader)*batch_size)

def compute_top1_and_top5_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float), top5_correct.astype(float)


def softmax_with_novel_prior(scores, novel_inds, base_inds, prior_m):
    scores = torch.exp(scores)
    scores_novel = scores[:, novel_inds]
    scores_base = scores[:, base_inds]
    tol = 0.0000001
    scores_novel *= prior_m / (tol + torch.sum(scores_novel, dim=1, keepdim=True).expand_as(scores_novel))
    scores_base *= (1.0 - prior_m) / (tol + torch.sum(scores_base, dim=1, keepdim=True).expand_as(scores_base))
    scores[:, novel_inds] = scores_novel
    scores[:, base_inds] = scores_base
    return scores
