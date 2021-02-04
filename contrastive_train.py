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
from Data.UCF101 import get_ucf101, get_ntuard
from utils import AverageMeter, accuracy
from models.contrastive_model import ContrastiveModel
import math
from torch.cuda.amp import autocast, GradScaler

DATASET_GETTERS = {'ucf101': get_ucf101, 'ntuard': get_ntuard}


def save_checkpoint(state, is_best, checkpoint):
    filename = f'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main_training_testing(EXP_NAME):
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Training')
    parser.add_argument('--out', default='results', help='directory to output the result')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--dataset', default='ucf101', type=str,
                        help='dataset name')
    parser.add_argument('--backbone', default='resnet3D18', type=str,
                        help='when contrastive architecture is selected, chose the backbone')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--frames-path', default='/datasets/UCF-101/Frames/frames-128x128/', type=str,
                        help='video frames path')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--no-clips', default=1, type=int,
                        help='number of clips')
    parser.add_argument('--feature-size', default=128, type=int,
                        help='size of feature embedding')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate, default 0.03')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='use pretrained version')
    parser.add_argument('--use-gru', action='store_true', default=False,
                        help='use gru')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--num-class', default=101, type=int,
                        help='total classes')

    args = parser.parse_args()
    print(args)
    EXP_NAME += str(args.backbone) + str(args.num_workers) + str(args.batch_size) + '_' + str(
        args.pretrained) + '_clips_' + str(args.no_clips) + '_gru_' + str(args.use_gru) + '_CS_' + str(
        args.cross_subject)
    print(EXP_NAME)
    out_dir = os.path.join(args.out, EXP_NAME)

    def create_model(args):
        if args.backbone == 'resnet3D18':
            import models.video_resnet as models
            model = models.r3d_18(num_classes=args.num_class, pretrained=args.pretrained)
        elif args.backbone == 'i3d':
            import models.i3d as models
            model = models.i3d(num_classes=args.num_class, use_gru=args.use_gru, pretrained=args.pretrained,
                               pretrained_path='./models/rgb_imagenet.pt')
        return model

    def _init_backbone(num_classes):
        args.num_class = num_classes
        print(args)
        return create_model(args)

    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.seed != -1:
        set_seed(args)

    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir)

    train_dataset, _ = DATASET_GETTERS[args.dataset]('Data', args.frames_path, contrastive=True, num_clips=args.no_clips)

    model = ContrastiveModel(_init_backbone, args.feature_size)
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

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov)

    args.total_steps = args.epochs * args.iteration
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup * args.iteration, args.total_steps)

    min_train_loss=math.inf
    train_loss_list=[]

    if args.resume:
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        out_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        min_train_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = start_epoch

    model.zero_grad()

    for epoch in range(args.start_epoch, args.epochs):

        train_loss = train(args, train_loader, model, optimizer, scheduler, epoch)
        writer.add_scalar('train/1.train_loss', train_loss, epoch)
        is_best = min_train_loss > train_loss
        min_train_loss = train_loss if is_best else min_train_loss

        # save checkpoint
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'loss': train_loss,
                'best_loss': min_train_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, out_dir)

        train_loss_list.append(min_train_loss)
    with open(f'results/{EXP_NAME}/score_logger.txt', 'a+') as ofile:
        ofile.write(
            f'Best Loss: {min_train_loss}\n')

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
    scaler = GradScaler() # rescale the loss to get gradients when using autocast
    for batch_idx, (inputs_x, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs_x.to(args.device)

        with autocast():
            logits_x = model(inputs)
            logits, labels = loss.normalized_temp_cross_entropy_loss(logits_x)
            labels = labels.to(args.device)
            loss = F.cross_entropy(logits, labels, reduction='mean')

        # Scales the loss, and calls backward()
        # to create scaled gradients
        scaler.scale(loss).backward()
        losses.update(loss.item())

        # Unscales gradients and calls
        # or skips optimizer.step() if nan or inf TODO:check if there is a callback if this happens/end train
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()
        scheduler.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_progress:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Top 1 Acc: {acc:.3f}\n".format(
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

if __name__ == '__main__':
    cudnn.benchmark = True
    EXP_NAME = 'NTUARD_CONTRASTIVE'
    main_training_testing(EXP_NAME)
