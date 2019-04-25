

from  __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
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
        one_situation = torch.ones_like(y_mlp).scatter_(1,torch.zeros_like(y_mlp[:,0]).fill_(i).unsqueeze(-1).long(),0)
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
        #images = torch.FloatTensor(images)
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
    mu,sig = per_image_normalize(images)
    return ((images - mu.unsqueeze(-1))/sig.unsqueeze(-1))
    #return per_image_normalize(images)


def train(FLAGS, FLAGS_STR, logger):
    mnist = read_data_sets(FLAGS.data_path, n_labeled=FLAGS.num_labeled, one_hot=True)
    dnn = models.MNISTDNN() 
    #supervised_loss  = nn.CrossEntropyLoss(reduction='none')
    supervised_loss  = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(dnn.parameters(), lr=FLAGS.lr)
    
    #reset accuracies after 100 batches
    #eval after every 500 batches
    
    total, total_correct, total_labelled = 0,0,0
    total_sloss, total_uloss = 0.0, 0.0
    
    if FLAGS.gpu:
        dnn = dnn.cuda()


    for batch_num in range(50000):
        images, labels = mnist.train.next_batch(FLAGS.batch_size)
        images = torch.FloatTensor(images)
        labels = torch.Tensor(labels) 

        images = train_transform(images)
        if FLAGS.gpu:
            images = images.cuda()
            labels = labels.cuda()

        #labels = torch.Tensor(labels)
        label_examples = labels.sum(dim=1)
        unlabel_examples = 1.0 - label_examples 
        
        y_ = labels
        
        y_mlp = dnn(images)
    
        sloss = supervised_loss(y_mlp,y_).sum(dim=1)
        uloss = semantic_loss(y_mlp)
        loss = (FLAGS.wt*uloss + sloss*label_examples).mean()
        step(dnn,loss,optimizer)
        
        correct_prediction = label_examples*((y_mlp.max(dim=1)[1] == y_.max(dim=1)[1]).float())
        total_correct += correct_prediction.sum().item()
        total_labelled += label_examples.sum().item()
        total_sloss += (sloss*label_examples).sum().item()
        total_uloss += uloss.sum().item()
        total += labels.shape[0]
        
    
   
        if batch_num % 500 == 0:
            with torch.no_grad():
                dnn.eval()
                images = torch.FloatTensor(mnist.test.images)
                labels = torch.Tensor(mnist.test.labels)
                if FLAGS.gpu:
                    images = images.cuda()
                    labels = labels.cuda() 
                images = test_transform(images)
                y_ = labels 
                y_mlp = dnn(images)
                correct_prediction = (y_mlp.max(dim=1)[1] == y_.max(dim=1)[1]).sum().item()
                #print('Step {} Test_Accuracy {}'.format(batch_num, correct_prediction/y_.shape[0]))
                 
                logger.info('{},{},{:.4f},{},{},{}'.format(batch_num,FLAGS_STR,round(correct_prediction/y_.shape[0],4),round(total_correct/total_labelled,4),round(total_sloss/total_labelled,4),round(total_uloss/total,4)))
            #
            dnn.train()
    
        if batch_num % 10000 == 0:
            print('{},{},{:.4f},{},{},{}'.format(batch_num,FLAGS_STR,round(correct_prediction/y_.shape[0],4),round(total_correct/total_labelled,4),round(total_sloss/total_labelled,4),round(total_uloss/total,4)))

        if batch_num % 100 == 0:
            #print('step {}, train_accuracy {}, train_loss {}, uloss {}'.format(batch_num, total_correct/total_labelled, total_sloss / total_labelled, total_uloss /total))
            total, total_correct, total_labelled = 0,0,0
            total_sloss, total_uloss = 0.0, 0.0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                                            default='mnist_data',
                                            help='Directory for storing input data')
    parser.add_argument('--num_labeled', type=int,
                                            help='Num of labeled examples provided for semi-supervised learning.',default = 100)
    parser.add_argument('--batch_size', type=int,default=8,
                                            help='Batch size for mini-batch Adams gradient descent.')
    parser.add_argument('--wt', type=float,help='semantic loss weight', default = 0.0005)
    parser.add_argument('--std', type=float,help='std dev of gaussian noise', default = 0.3)
    parser.add_argument('--lr', type=float,help='learning rate of adam', default = 0.0001)
    parser.add_argument('--gpu', type=int,help='use gpu or not', default = 0)
     
    
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.gpu = torch.cuda.is_available() and FLAGS.gpu
    #print(yatin)
    keys = list(FLAGS.__dict__.keys())
    keys.sort()
    keys.remove('gpu')
    #Pdb().set_trace()
    FLAGS_STR = '_'.join([k.replace('_','.') +'-'+str(FLAGS.__dict__[k]) for k in keys])
    print('Start: {}'.format(FLAGS_STR))
    if os.path.exists('../logs_torch/'+FLAGS_STR+'.csv'):
        print('Alredy done. Exit')
    else:
        logger = logging.getLogger(FLAGS_STR)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s,%(message)s',datefmt='%Y%m%d %H:%M:%S')
        handler = logging.FileHandler('../logs_torch/'+FLAGS_STR+'.csv')
        logger.addHandler(handler)
        logger.info('t,step,exp,tea,tra,trl,trw')
        handler.setFormatter(formatter)
        train(FLAGS,FLAGS_STR,logger)
        #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
        print('End: {}'.format(FLAGS_STR))

