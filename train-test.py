import argparse
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
from collections import OrderedDict
import pickle 
import scipy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Data.UCF101 import get_ucf101
from utils import AverageMeter, accuracy

DATASET_GETTERS = {'ucf101': get_ucf101}

def save_checkpoint(state, is_best, checkpoint):
    filename=f'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,f'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main_training_testing(EXP_NAME):
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--out', default=f'results/{EXP_NAME}', help='directory to output the result')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--dataset', default='ucf101', type=str,
                        help='dataset name')
    parser.add_argument('--arch', default='resnet3D18', type=str,
                        help='dataset name')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--frames-path', default='/datasets/UCF-101/Frames/frames-128x128/', type=str,
                        help='video frames path')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate, default 0.03')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--num-class', default=101, type=int,
                        help='total classes')

    args = parser.parse_args()
    best_acc = 0
    best_acc_2 = 0

    def create_model(args):
        if args.arch == 'resnet3D18':
            import models.video_resnet as models
            model = models.r3d_18(num_classes=args.num_class)

        return model
    
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.seed != -1:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(args.out)

    train_dataset, test_dataset = DATASET_GETTERS[args.dataset]('Data', args.frames_path)

    model = create_model(args)
    model.to(args.device)

    args.iteration = len(train_dataset) // args.batch_size // args.world_size
    train_sampler = RandomSampler 

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    
    args.total_steps = args.epochs * args.iteration
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup * args.iteration, args.total_steps)

    if args.resume:
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = start_epoch

    test_accs = []
    model.zero_grad()
    
    for epoch in range(args.start_epoch, args.epochs):
     
        train_loss = train(args, train_loader, model, optimizer, scheduler, epoch)
        test_loss = 0.0
        test_acc = 0.0
        test_acc_2 = 0.0
        test_model = model

        test_loss, test_acc, test_acc_2 = test(args, test_loader, test_model, epoch)

        if epoch > (args.epochs+1)/2 and epoch%30==0: 
            test_loss, test_acc, test_acc_2 = test(args, test_loader, test_model, epoch)
        elif epoch == (args.epochs-1):
            test_loss, test_acc, test_acc_2 = test(args, test_loader, test_model, epoch)

        writer.add_scalar('train/1.train_loss', train_loss, epoch)
        writer.add_scalar('test/1.test_acc', test_acc, epoch)
        writer.add_scalar('test/2.test_loss', test_loss, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        is_best_2 = test_acc_2 > best_acc_2
        best_acc_2 = max(test_acc_2, best_acc_2)

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

        test_accs.append(test_acc)
    with open(f'results/{EXP_NAME}/score_logger.txt', 'a+') as ofile:
        ofile.write(f'Last Acc (after softmax): {test_acc}, Best Acc (after softmax): {best_acc}, Last Acc (before softmax): {test_acc_2}, Best Acc (before softmax): {best_acc_2}\n')

    if args.local_rank in [-1, 0]:
        writer.close()



def train(args, labeled_trainloader, model, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = labeled_trainloader
    model.train()
    for batch_idx, (inputs_x, targets_x) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs_x.to(args.device)
        targets_x = targets_x.to(args.device)
        
        logits_x = model(inputs)
        loss = F.cross_entropy(logits_x, targets_x, reduction='mean')
        loss.backward()
        losses.update(loss.item())

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_progress:
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=scheduler.get_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()
    
    return losses.avg


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    predicted_target = {}
    ground_truth_target = {}
    predicted_target_not_softmax = {}

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets, video_name) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            out_prob = F.softmax(outputs, dim=1)
            out_prob = out_prob.cpu().numpy().tolist()
            targets = targets.cpu().numpy().tolist()
            outputs = outputs.cpu().numpy().tolist()
            
            for iterator in range(len(video_name)):
                if video_name[iterator] not in predicted_target:
                    predicted_target[video_name[iterator]] = []
                
                if video_name[iterator] not in predicted_target_not_softmax:
                    predicted_target_not_softmax[video_name[iterator]] = []

                if video_name[iterator] not in ground_truth_target:
                    ground_truth_target[video_name[iterator]] = []

                predicted_target[video_name[iterator]].append(out_prob[iterator])
                predicted_target_not_softmax[video_name[iterator]].append(outputs[iterator])
                ground_truth_target[video_name[iterator]].append(targets[iterator])
                
            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg
                ))
        if not args.no_progress:
            test_loader.close()

    for key in predicted_target:
        clip_values = np.array(predicted_target[key]).mean(axis=0)
        video_pred = np.argmax(clip_values)
        predicted_target[key] = video_pred
    
    for key in predicted_target_not_softmax:
        clip_values = np.array(predicted_target_not_softmax[key]).mean(axis=0)
        video_pred = np.argmax(clip_values)
        predicted_target_not_softmax[key] = video_pred
    
    for key in ground_truth_target:
        clip_values = np.array(ground_truth_target[key]).mean(axis=0)
        ground_truth_target[key] = int(clip_values)

    pred_values = []
    pred_values_not_softmax = []
    target_values = []

    for key in predicted_target:
        pred_values.append(predicted_target[key])
        pred_values_not_softmax.append(predicted_target_not_softmax[key])
        target_values.append(ground_truth_target[key])
    
    pred_values = np.array(pred_values)
    pred_values_not_softmax = np.array(pred_values_not_softmax)
    target_values = np.array(target_values)

    secondary_accuracy = (pred_values == target_values)*1
    secondary_accuracy = (sum(secondary_accuracy)/len(secondary_accuracy))*100
    print(f'test accuracy after softmax: {secondary_accuracy}')

    secondary_accuracy_not_softmax = (pred_values_not_softmax == target_values)*1
    secondary_accuracy_not_softmax = (sum(secondary_accuracy_not_softmax)/len(secondary_accuracy_not_softmax))*100
    print(f'test accuracy before softmax: {secondary_accuracy_not_softmax}')

    return losses.avg, secondary_accuracy, secondary_accuracy_not_softmax


if __name__ == '__main__':
    cudnn.benchmark = True
    EXP_NAME = 'UCF101_SUPERVISED_TRAINING'
    main_training_testing(EXP_NAME)
