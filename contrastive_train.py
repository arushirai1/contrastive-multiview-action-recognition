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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Data.UCF101 import get_ucf101, get_ntuard
from utils import AverageMeter, accuracy
from models.contrastive_model import ContrastiveModel, ContrastiveMultiTaskModel
import math
from loss import info_nce_loss
from torch.cuda.amp import autocast, GradScaler
import einops

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return tuple(res)

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
    parser.add_argument('--no-views', default=2, type=int,
                        help='number of views (positives+1)')
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
    parser.add_argument('--multiview', action='store_true', default=False,
                        help='use multiview version')
    parser.add_argument('--use-gru', action='store_true', default=False,
                        help='use gru')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='use augmentations defined in simclr')
    parser.add_argument('--hard-positive', action='store_true', default=False,
                        help='use hard positives')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--cross-subject', action='store_true', default=False,
                        help='Training and testing on cross subject split')
    parser.add_argument('--random_temporal', action='store_true', default=False,
                        help='Training and testing on cross subject split')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--num-class', default=101, type=int,
                        help='total classes')
    parser.add_argument('--exp-name', default='NTUARD_SUPERVISED_TRAINING', type=str,
                        help='Experiment name')
    parser.add_argument('--semi-supervised-contrastive-joint-training', action='store_true', default=False,
                        help='Will train supervised and contrastive model simultaneously')
    parser.add_argument('--percentage', default=1.0, type=float,
                        help='in semi-supervised setting, what split do you want to use')

    args = parser.parse_args()
    if args.cross_subject and args.no_views < 3 and not args.hard_positive:
        args.no_views = 3
    print(args)
    EXP_NAME = str(args.exp_name) + str(args.backbone) + str(args.num_workers) + str(args.batch_size) + '_' + str(
        args.pretrained) + '_clips_' + str(args.no_clips) + '_lr_' + str(args.lr)+'augment'+str(args.augment)
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
        return create_model(args)

    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.seed != -1:
        set_seed(args)

    os.makedirs(out_dir, exist_ok=True)
    # remove previous logs
    os.system(f'rm {out_dir}/events.*')

    writer = SummaryWriter(out_dir)

    train_datasets, test_dataset = DATASET_GETTERS[args.dataset]('Data', args.frames_path, contrastive=True, num_clips=args.no_clips, multiview=args.multiview, augment=args.augment, cross_subject=args.cross_subject, hard_positive=args.hard_positive, random_temporal=args.random_temporal)

    if args.semi_supervised_contrastive_joint_training:
        with train_datasets[-1] as train_dataset:
            train_datasets[-1] = random_split(train_dataset, (round(args.percentage * len(train_dataset)), round((1 - args.percentage) * len(train_dataset))))[0]
        model = ContrastiveMultiTaskModel(_init_backbone, args.feature_size, args.num_classes)
    else:
        model = ContrastiveModel(_init_backbone, args.feature_size)
    args.iteration = sum([len(dataset) for dataset in train_datasets]) // args.batch_size // args.world_size
    model.to(args.device)

    train_loaders = [DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True) for train_dataset in train_datasets]

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

    min_train_loss=math.inf
    best_acc=0
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
    try:
        for epoch in range(args.start_epoch, args.epochs):

            train_losses, train_accs = train(args, train_loaders, model, optimizer, scheduler, epoch)
            for mode in ["contrastive", "semi_supervised"]:
                writer.add_scalar(mode+'Loss/train', train_losses[mode], epoch)
                writer.add_scalar(mode+'Accuracy/train', train_accs[mode], epoch)
            if args.semi_supervised_contrastive_joint_training:
                test_loss, test_acc, _, _ = test(args, test_loader, model, epoch)
                writer.add_scalar('Loss/test', test_loss, epoch)
                writer.add_scalar('Accuracy/test', test_acc, epoch)
                is_best = test_acc > best_acc
                best_acc = test_acc if is_best else best_acc
            else:
                is_best = min_train_loss > train_losses["contrastive"]
                min_train_loss = train_losses["contrastive"] if is_best else min_train_loss

            # save checkpoint
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'loss': train_losses["contrastive"],
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
    except Exception as e:
        print("Exception", e)

def train(args, train_loaders, model, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = {}
    top5 = AverageMeter()
    losses = {}
    end = time.time()
    # TODO each mode should have seperate batch size

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    model.train()
    scaler = GradScaler() # rescale the loss to get gradients when using autocast
    for mode, train_loader in zip(["contrastive", "semi_supervised"], train_loaders):
        losses[mode] = AverageMeter()  # seperate loss for each mode
        top1[mode] = AverageMeter()  # seperate acc for each mode
        for batch_idx, (inputs_x, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            if mode == "contrastive":
                inputs_x = torch.cat(inputs_x, dim=0)
                labels = torch.cat([labels for i in range(args.no_views)])

            inputs = inputs_x.to(args.device)
            labels = labels.to(args.device)

            with autocast():
                logits_x = model(inputs, mode)
                '''
                try:
                    logits, labels = info_nce_loss(logits_x, args.batch_size, args.no_views, supervised=args.hard_positive, labels=labels)
                except Exception as e:
                '''
                if mode == "contrastive":
                    logits, labels = info_nce_loss(logits_x, args.batch_size, args.no_views)
            labels = labels.type(torch.LongTensor).to(args.device)
            loss = F.cross_entropy(logits, labels, reduction='mean')
            batch_top1, batch_top5 = accuracy(logits, labels, topk=(1, 5))

            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()
            losses[mode].update(loss.item())
            top1[mode].update(batch_top1[0])
            top5.update(batch_top5[0])
            # Unscales gradients and calls
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()
            scheduler.step()
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Top 1 Acc: {acc:.3f} Mode: {mode}".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.iteration,
                        lr=scheduler.get_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses[mode].avg,
                        acc=top1[mode].avg,
                        mode=mode))
                p_bar.update()
    if not args.no_progress:
        p_bar.close()

    return {mode:loss.avg for mode, loss in losses.items()}, {mode:acc.avg for mode, acc in top1.items()}

def test(args, test_loader, model, eval_mode=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    if not args.no_progress:
        test_loader = tqdm(test_loader)

    results={i:[] for i in range(args.num_class)}
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if eval_mode:
                inputs=einops.rearrange(inputs, 'b c ... -> (b c) ...', c = args.no_clips)
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            if eval_mode:
                outputs=einops.reduce(outputs, '(b c) logits -> b logits', 'mean', c = args.no_clips)

            loss = F.cross_entropy(outputs, targets, reduction='mean')

            for i, target in enumerate(targets):
                results[target.item()].append(np.argmax(outputs.data[i].cpu().numpy()))

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Top 1 Acc: {acc:.3f}".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        acc=top1.avg
                    ))
        if not args.no_progress:
            test_loader.close()

    return losses.avg, top1.avg, top5.avg, results

if __name__ == '__main__':
    cudnn.benchmark = True
    EXP_NAME = 'NTUARD_CONTRASTIVE'
    main_training_testing(EXP_NAME)
