

from  __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np 
from mnist_input import read_data_sets
import argparse
import sys
import tempfile
import models
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision as tv 
FLAGS = None


def semantic_loss(y_mlp):
    prob = F.sigmoid(y_mlp)
    wmc_tmp = torch.zeros_like(prob)
    for i in range(y_mlp.shape[1]):
        one_situation = torch.ones_like(y_mlp).scatter_(1,torch.LongTensor(y_mlp.shape[0]).fill_(i).unsqueeze(-1),0)
        wmc_tmp[:,i] = torch.abs((one_situation - prob).prod(dim=1))
        #omp = 1.0 - prob
        #omp[:,i] = prob[:,i]
        #wmc_tmp[:,i] = omp.prod(dim=1)

    wmc_tmp = -1.0*torch.log(wmc_tmp.sum(dim=1))
    return wmc_tmp


def step(model, loss, optimizer, clip_val=None):
    optimizer.zero_grad()
    loss.backward()
    # clip gradients
    if clip_val is not None:
        torch.nn.utils.clip_grad_norm(model.parameters(), clip_val)
    optimizer.step()

def per_image_normalize(images):
    #images = torch.FloatTensor(images)
    mu = images.mean(dim=1)
    sig = images.std(dim=1)
    sig = sig.clamp(min=1.0/math.sqrt(images.shape[1]))
    return mu,sig 
    #return ((images - mu.unsqueeze(-1))/sig.unsqueeze(-1))
    

def train_transform(images):
    with torch.no_grad():
        images = torch.FloatTensor(images)
        mu,sig = per_image_normalize(images)
       
        #all of this is just to crop and pad with 0
        to_pil = tv.transforms.ToPILImage()
        to_tensor = tv.transforms.ToTensor()
        rc = tv.transforms.RandomCrop(24)

        images = torch.cat([to_tensor(rc(to_pil(ten.view(28,28)))) for ten in images])
        
        images = F.pad(((images.view(-1,24*24) - mu.unsqueeze(-1))/sig.unsqueeze(-1)).view(-1,24,24),(2,2,2,2)).view(-1,784)
        
        images += 0.3*torch.randn(images.size())
    return images  


def test_transform(images):
    images = torch.FloatTensor(images)
    mu,sig = per_image_normalize(images)
    return ((images - mu.unsqueeze(-1))/sig.unsqueeze(-1))
    #return per_image_normalize(images)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                default='/home/yatin/phd/fashion-mnist/data/fashion',help='Directory for storing input data')
parser.add_argument('--num_labeled', type=int,default = 100,
                    help='Num of labeled examples provided for semi-supervised learning.')
parser.add_argument('--batch_size', type=int,default = 10,
             help='Batch size for mini-batch Adams gradient descent.')

parser.add_argument('--lr', type=float,default = 0.002,help='lr for adam')
parser.add_argument('--lamda', type=float,default = 0.0005,help='constant wt for semantic loss')

FLAGS, unparsed = parser.parse_known_args()


mnist = read_data_sets('/home/yatin/phd/fashion-mnist/data/fashion',100,one_hot = True)


dnn = models.MNISTDNN() 
#supervised_loss  = nn.CrossEntropyLoss(reduction='none')
supervised_loss  = nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.Adam(dnn.parameters(), lr=FLAGS.lr)

#reset accuracies after 100 batches
#eval after every 500 batches

total, total_correct, total_labelled = 0,0,0
total_sloss, total_uloss = 0.0, 0.0

for batch_num in range(50000):
    images, labels = mnist.train.next_batch(FLAGS.batch_size)
    #break
    images = train_transform(images)
    labels = torch.Tensor(labels)
    label_examples = labels.sum(dim=1)
    unlabel_examples = 1.0 - label_examples 
    
    #y_ = labels.max(dim=1)[1]
    y_ = labels
    
    y_mlp = dnn(images)
    #break

    sloss = supervised_loss(y_mlp,y_).sum(dim=1)
    uloss = semantic_loss(y_mlp)
    loss = (FLAGS.lamda*uloss + sloss*label_examples).mean()
    step(dnn,loss,optimizer)
    
    correct_prediction = label_examples*((y_mlp.max(dim=1)[1] == y_.max(dim=1)[1]).float())
    total_correct += correct_prediction.sum().item()
    total_labelled += label_examples.sum().item()
    total_sloss += (sloss*label_examples).sum().item()
    total_uloss += uloss.sum().item()
    total += labels.shape[0]
    
    if batch_num % 100 == 0:
        print('step {}, train_accuracy {}, train_loss {}, uloss {}'.format(batch_num, total_correct/total_labelled, total_sloss / total_labelled, total_uloss /total))
    
    if batch_num % 500 == 0:
        with torch.no_grad():
            dnn.eval()
            images = mnist.test.images
            labels = mnist.test.labels
            images = test_transform(images)
            y_ = torch.Tensor(labels)
            y_mlp = dnn(images)
            correct_prediction = (y_mlp.max(dim=1)[1] == y_.max(dim=1)[1]).sum().item()
            print('Step {} Test_Accuracy {}'.format(batch_num, correct_prediction/y_.shape[0]))
        #
        dnn.train()

