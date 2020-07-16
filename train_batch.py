import sys
import os

import warnings

from model import *

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import loading_data

parser = argparse.ArgumentParser(description='PyTorch VGG bn')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

def main():
    
    global args,best_prec1, writer
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-6
    args.lr = 1e-5
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [0, 200]
    args.scales        = [1, 0.1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 100
    args.log_path = './train_log/baseline_bn_dil'
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = baseline_bn(dilation=True)
    
    model = model.cuda()
    
    criterion = nn.MSELoss().cuda()
    
    '''optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)'''
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=args.log_path)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    for epoch in range(args.start_epoch, args.epochs):
        
        #adjust_learning_rate(optimizer, epoch)
        #writer.add_scalar('lr',args.lr, epoch)
        train(train_list, model, criterion, optimizer, epoch)
        print('check model on training set...')
        mae = validate(train_list, model, criterion)
        writer.add_scalar('MAE_train', mae, epoch)
        print('check model on validation set...')
        prec1 = validate(val_list, model, criterion)
        writer.add_scalar('MAE', prec1, epoch)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)
    writer.close()
    final_test(val_list, model)

def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    

    train_loader = loading_data.loading_train_data(train_list, batch_size=8)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model = nn.DataParallel(model, device_ids=[0,1])
    model.train()
    end = time.time()
    with tqdm(total=len(train_loader)) as pbar:
        for i,(img, target)in enumerate(train_loader):
            data_time.update(time.time() - end)

            img = img.cuda()
            img = Variable(img)
            output = model(img)

            target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
            target = Variable(target)

            loss = criterion(output, target)

            losses.update(loss.item(), img.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                tqdm.write('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      .format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))
            pbar.set_description('iterations')
            pbar.set_postfix(ordered_dict={'epoch': epoch})
            pbar.update(1)
        writer.add_scalar('loss_avg', losses.avg, epoch)
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)
    
    model.eval()
    
    mae = 0
    with torch.no_grad():
        for i,(img, target) in enumerate(test_loader):
            img = img.cuda()
            img = Variable(img)
            output = model(img)

            mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae

def final_test(test_list, model):
    print ('begin final test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(test_list,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    model.load_state_dict(torch.load('./saved_models/'+args.task+'model_best.pth.tar')['state_dict'])
    
    mae = 0
    mse = 0
    with torch.no_grad():
        for i,(img, target) in enumerate(test_loader):
            img = img.cuda()
            img = Variable(img)
            output = model(img)

            item =  abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
            mae += item
            mse += item*item
        
    mae = mae/len(test_loader)
    mse = torch.sqrt(mse/len(test_loader))
    print('MAE:', mae, 'MSE:', mse)
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    #args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        if epoch == args.steps[i]:

            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.scales[i]
            break
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        
