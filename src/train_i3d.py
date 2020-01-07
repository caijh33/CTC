import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import sys
import argparse
from utils import Timer
from utils import Logger
from utils import MixUp
import shutil
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('--save_model', default= 'checkpoints/', type=str)
parser.add_argument('--arch',default='i3d', type=str, choices=['i3d', 'se_i3d', 'bilinear_i3d', 'se_bilinear_i3d'])
parser.add_argument('--dataset',default='hmdb51', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('--root', type=str)
parser.add_argument('--train_list', default='data/hmdb51/hmdb51_rgb_train_split_1.txt', type=str)
parser.add_argument('--val_list', default='data/hmdb51/hmdb51_rgb_val_split_1.txt', type=str)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--dropout', '--do', default=0.36, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[10,20,25,30,35,40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-7)')

parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')


parser.add_argument('--snapshot_pref', type=str, default="")

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np

#from pytorch_i3d import InceptionI3d, get_fine_tuning_parameters

#from charades_dataset import Charades as Dataset

from dataset.ucf101_dataset import I3dDataSet
from net.basic_models.weight_transform import weight_transform
#from net.basic_models.focal_loss import FocalLoss
#add for kinetics
#from temporal_transforms import LoopPadding, TemporalRandomCrop
#from target_transforms import ClassLabel, VideoID

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 250


_CHECKPOINT_PATHS = {
    'rgb': 'pretrained_models/rgb_scratch.pkl',
    'flow': 'pretrained_models/flow_scratch.pkl',
    'rgb_imagenet': 'pretrained_models/rgb_imagenet.pkl',
    'flow_imagenet': 'pretrained_models/flow_imagenet.pkl',
}

class DisturbLabel(torch.nn.Module):
    def __init__(self, alpha, C):
        super(DisturbLabel, self).__init__()
        self.alpha = alpha
        self.C = C
        # Multinoulli distribution
        self.p_c = (1 - ((C - 1)/C) * (alpha/100))
        self.p_i = (1 / C) * (alpha / 100)

    def forward(self, y):
        # convert classes to index
        y_tensor = y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)

        # create disturbed labels
        depth = self.C
        y_one_hot = torch.ones(y_tensor.size()[0], depth) * self.p_i
        y_one_hot.scatter_(1, y_tensor, self.p_c)
        y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))

        # sample from Multinoulli distribution
        distribution = torch.distributions.OneHotCategorical(y_one_hot)
        y_disturbed = distribution.sample()
        y_disturbed = y_disturbed.max(dim=1)[1]  # back to categorical

        return y_disturbed

def main():
    best_prec1 = 0
    with open('logs/' + args.dataset +'/' + args.arch + '_'  + args.mode + '_validation.txt', 'a') as f:
        f.write("=============================================")
        f.write('\n')
        f.write("lr: ")
        f.write(str(args.lr))
        f.write(" lr_step: ")
        f.write(str(args.lr_steps))
        f.write(" dataset: ")
        f.write(str(args.dataset))
        f.write(" modality: ")
        f.write(str(args.mode))
        f.write(" dropout: ")
        f.write(str(args.dropout))
        f.write(" batch size: ")
        f.write(str(args.batch_size))
        f.write('\n')
    if args.dataset == 'ucf101':
        num_class = 101
        data_length = 64
        image_tmpl = "frame{:06d}.jpg"
    elif args.dataset == 'hmdb51':
        num_class = 51
        data_length = 64
        image_tmpl = "img_{:05d}.jpg"
        #opt.temporal_train_data_root = "/home/wjp/Data/tvl1_flow/"
    elif args.dataset == 'kinetics':
        num_class = 400
        data_length = 64
        image_tmpl = "img_{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    val_logger = Logger('logs/' + args.dataset + '/' + args.arch + '_'  + args.mode + '_val.log', ['epoch', 'acc'])
    # define loss function (criterion) and optimizer
    #======================data transform=============

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    #=======================design the dataset==============
    train_dataset = I3dDataSet("", args.train_list, num_segments=1,
                   new_length=data_length,
                   modality=args.mode,
                   dataset = args.dataset,
                   image_tmpl=image_tmpl if args.mode in ["rgb", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=train_transforms,
                   test_mode=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = I3dDataSet("", args.val_list, num_segments=1,
                   new_length=data_length,
                   modality=args.mode,
                   dataset = args.dataset,
                   image_tmpl=image_tmpl if args.mode in ["rgb", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=test_transforms,
                   test_mode=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    datasets = {'train': train_dataset, 'val': val_dataset}

    #=============================set the model ==================
        # setup the model
    if args.mode == 'flow':
        #i3d = InceptionI3d(400, in_channels=3)
        if args.arch == 'i3d':
            from net.i3d import I3D
            i3d = I3D(modality='flow', num_classes = num_class, dropout_prob = args.dropout)
        elif args.arch == 'bilinear_i3d':
            from net.bilinear_i3d import I3D
            i3d = I3D(modality='flow', num_classes = num_class, dropout_prob = args.dropout)
        elif args.arch == 'se_i3d':
            from net.se_i3d import I3D
            i3d = I3D(modality='flow', num_classes = num_class, dropout_prob = args.dropout)
        elif args.arch == 'se_bilinear_i3d':
            from net.se_bilinear_i3d import I3D
            i3d = I3D(modality='flow', num_classes = num_class, dropout_prob = args.dropout)
        else:
            Exception("not support now!")
        i3d.eval()
        pretrain_dict = torch.load('pretrained_models/model_flow.pth')
        #pretrain_dict = torch.load(_CHECKPOINT_PATHS['rgb_imagenet'])
        model_dict = i3d.state_dict()
        #pretrain_dict = torch.load('pretrained_models/rgb_imagenet.pt')
        #weight_dict = {k:v for k, v in pretrain_dict.items() if (k in model_dict)}
        weight_dict = weight_transform(model_dict, pretrain_dict)
        #model.load_state_dict(weight_dict)
        # model_dict.update(weight_dict)
        i3d.load_state_dict(weight_dict)
    else:
        #i3d = InceptionI3d(400, in_channels=3)
        if args.arch == 'i3d':
            from net.i3d import I3D
            i3d = I3D(modality='rgb', num_classes = num_class, dropout_prob = args.dropout)
        elif args.arch == 'se_i3d':
            from net.se_i3d import I3D
            i3d = I3D(modality='rgb', num_classes = num_class, dropout_prob=args.dropout)
        elif args.arch == 'bilinear_i3d':
            from net.bilinear_i3d import I3D
            i3d = I3D(modality='rgb', num_classes = num_class, dropout_prob = args.dropout)
        elif args.arch == 'se_bilinear_i3d':
            from net.se_bilinear_i3d import I3D
            i3d = I3D(modality='rgb', num_classes = num_class, dropout_prob=args.dropout)
        else:
            Exception("not support now!")
        i3d.eval()
        pretrain_dict = torch.load('pretrained_models/model_rgb.pth')
        #pretrain_dict = torch.load(_CHECKPOINT_PATHS['rgb_imagenet'])
        model_dict = i3d.state_dict()
        #pretrain_dict = torch.load('pretrained_models/rgb_imagenet.pt')
        #weight_dict = {k:v for k, v in pretrain_dict.items() if (k in model_dict)}
        weight_dict = weight_transform(model_dict, pretrain_dict)
        #model.load_state_dict(weight_dict)
        # model_dict.update(weight_dict)
        i3d.load_state_dict(weight_dict)
        #i3d.load_state_dict(torch.load('pretrained_models/rgb_imagenet.pt'))
    #if num_class != 400:
    #    i3d.replace_logits(num_class)
    #i3d.load_state_dict(torch.load('/ssd/net/000920.pt'))
    i3d.cuda()
    #print(i3d)
    #============================set SGD, critization and lr ==================
    optimizer = torch.optim.SGD(i3d.parameters(),
                                lr = args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                dampening=0,
                                nesterov=False)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience = 3)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    model = nn.DataParallel(i3d)
    criterion = torch.nn.NLLLoss().cuda()
    disturb = DisturbLabel(alpha= 10, C=51)
    # criterion = FocalLoss(gamma = 0).cuda()
    #print(model)

    writer = SummaryWriter() #create log folders for plot
    timer = Timer()
    for epoch in range(1, args.epochs):
        timer.tic()
        #scheduler.step()
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        #prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

        # train for one epoch
        train_prec1, train_loss = train(train_loader, model, criterion, optimizer, epoch,disturb)
        writer.add_scalar('Train/Accu', train_prec1, epoch)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, val_loss = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))
            writer.add_scalar('Val/Accu', prec1, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalars('data/Acc', {'train_prec1': train_prec1, 'val_prec1':prec1}, epoch)
            writer.add_scalars('data/Loss', {'train_loss': train_loss, 'val_loss':val_loss}, epoch)
            #scheduler.step(val_loss)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, best_prec1)
            val_logger.log({
                'epoch': epoch,
                'acc': prec1
            })
        timer.toc()
        left_time = timer.average_time * (args.epochs - epoch)
        print("best_prec1 is: {}".format(best_prec1))
        print("left time is: {}".format(timer.format(left_time)))
        with open('logs/' + args.dataset + '/' + args.arch + '_'  + args.mode + '_validation.txt', 'a') as f:
            f.write(str(epoch))
            f.write(" ")
            f.write(str(train_prec1))
            f.write(" ")
            f.write(str(prec1))
            f.write(" ")
            f.write(timer.format(timer.diff))
            f.write('\n')
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, disturb):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mixup_deteor = MixUp(1)
    # switch to train mode
    #model.train()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        #input, targer_a, tar
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        target = disturb(target)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], first_lr:{first_lr:.7f} last_lr: {last_lr:.7f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5,
                   first_lr=optimizer.param_groups[0]['lr'],
                   last_lr=optimizer.param_groups[-1]['lr'])))
    return top1.avg, losses.avg

'''
def train(train_loader, model, criterion, optimizer, epoch, disturb):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mixup_deteor = MixUp(1)
    # switch to train mode
    #model.train()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        #input, targer_a, tar
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        # target = disturb(target)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var = disturb(target_var)
        # compute output
        output = model(input_var)
        #print(output)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], first_lr:{first_lr:.7f} last_lr: {last_lr:.7f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5,
                   first_lr=optimizer.param_groups[0]['lr'],
                   last_lr=optimizer.param_groups[-1]['lr'])))
    return top1.avg, losses.avg
'''

def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #model.train(False)
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))

            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))
    #loss.data.cpu().numpy()
    #loss.data[0]
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, prec1, filename='_checkpoint.pth.tar'):
    filename = args.save_model + args.dataset  + '/'+ args.mode + filename
    torch.save(state, filename)
    if is_best and prec1 > 70:
        best_name = args.save_model + args.dataset + '/' + str(prec1)[0:6] + '_' + args.mode + '_model_best.pth.tar'
        shutil.copyfile(filename, best_name)


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

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
