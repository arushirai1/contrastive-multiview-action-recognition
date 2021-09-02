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
import einops
from models.generator import get_decoder
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


def main_training_testing():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--out', default='results', help='directory to output the result')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--dataset', default='ucf101', type=str,
                        help='dataset name')
    parser.add_argument('--arch', default='resnet3D18', type=str,
                        help='model name')
    parser.add_argument('--exp-name', default='NTUARD_SUPERVISED_TRAINING', type=str,
                        help='Experiment name')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--frames-path', default='/datasets/UCF-101/Frames/frames-128x128/', type=str,
                        help='video frames path')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--percentage', default=1.0, type=float,
                        help='in semi-supervised setting, what split do you want to use')
    parser.add_argument('--no-clips', default=1, type=int,
                        help='number of clips')
    parser.add_argument('--no-frames', default=8, type=int,
                        help='number of clips')
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
    parser.add_argument('--spatio-temporal', action='store_true', default=False,
                        help='Flatten in the last layer')
    parser.add_argument('--feature-size', default=128, type=int,
                        help='size of feature embedding')
    parser.add_argument('--cross-subject', action='store_true', default=False,
                        help='Training and testing on cross subject split')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--num-class', default=101, type=int,
                        help='total classes')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='use augmentations defined in simclr')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='select method of eval: finetune (True) or linear probe (False)')
    parser.add_argument('--endpoint', default='A', type=str,
                        help='the layer the representation is extracted from from')
    parser.add_argument('--d-model', default=128, type=int,
                        help='Dimension of the hidden representations used in the transformer head')
    parser.add_argument('--num-layers', default=3, type=int,
                        help='Number of encoder layers in the transformer head')
    parser.add_argument('--num-heads', default=2, type=int,
                        help='Number of attention heads in each encoder layer')
    parser.add_argument('--base_endpoint', default='fc', type=str,
                        help='the endpoint of the base model before the transformer')
    parser.add_argument('--contrastive-mod', default='', type=str,
                        help='modification on traditional contrastive architecture')
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help='Eval only mode')
    parser.add_argument('--pos', action='store_true', default=False,
                        help='add positional embedding for transformers')
    parser.add_argument('--decoder', action='store_true', default=False,
                        help='Will add the decoder') # TODO remove this option
    parser.add_argument('--cnndecoder', action='store_true', default=False,
                        help='Will add the cnndecoder') # TODO remove this option
    parser.add_argument('--local-contrastive-loss', action='store_true', default=False,
                        help='Will add the local contrastive loss') # TODO remove this option
    parser.add_argument('--normalize-constant', default=1, type=int,
                        help='Divide values by') # TODO remove this option
    parser.add_argument('--simple', action='store_true', default=False,
                        help='Will add just one block') # TODO remove this option
    parser.add_argument('--sigmoid-loss', action='store_true', default=False,
                        help='Use sigmoid loss for reconstruction instead of MSE ') # TODO remove this option
    parser.add_argument('--sigmoid-activation', action='store_true', default=False,
                        help='Use sigmoid activation for reconstruction ') # TODO remove this option
    parser.add_argument('--joint-only-multiview-training', action='store_true', default=False,
                        help='train using joint view data')
    parser.add_argument('--combined-multiview-training', action='store_true', default=False,
                        help='train using joint view and single view data')
    parser.add_argument('--curriculum-learning', action='store_true', default=False,
                        help='have a graduated learning scheme where initially single views are presented and then multiview joint data is presented after a number of epochs')
    parser.add_argument('--pretrained-path', default='', type=str, help='path to weights of contrastive pretrained model')
    parser.add_argument('--tcl', action='store_true', default=False,
                        help='train using tcl')
    parser.add_argument('--no-delete', action='store_true', default=False,
                        help='don\'t delete contents of folder')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='shuffle frames')
    parser.add_argument('--mean', default=128., type=float,
                        help='mean')
    parser.add_argument('--std', default=128., type=float,
                        help='std')
    parser.add_argument('--temporally-consistent-spatial-augment', action='store_true', default=False,
                        help='Will use temporal consistent spatial augmentations')

    args = parser.parse_args()
    print(args)

    if args.arch == 'contrastive':
        EXP_NAME = args.exp_name + str(args.arch) + "_endpoint_" + str(args.endpoint) + "_finetune_" + str(args.finetune) + str(args.cross_subject)
    else:
        EXP_NAME = args.exp_name + str(args.arch) + str(args.num_workers) + str(args.batch_size) + '_' + str(
            args.pretrained) + '_clips_' + str(args.no_clips) + '_gru_' + str(args.use_gru) + '_CS_' + str(
            args.cross_subject)

    print(EXP_NAME)
    out_dir = os.path.join(args.out, EXP_NAME)
    best_acc = 0

    def init_transformer(args):
        # only supporting resnet3d, TODO: Add support for i3d
        from models import video_resnet
        base_model = video_resnet.r3d_18(endpoint=args.base_endpoint, spatio_temporal = args.spatio_temporal, positional_flag = int(args.pos))

        if args.arch == 'transformer':
            from models import transformer_model2 as transformer_model
            model = transformer_model.TransformerModel(base_model, args.num_class, d_model=args.d_model, N=args.num_layers, h=args.num_heads, dropout=0.3, endpoint=args.base_endpoint, positional_flag = args.pos, eval_mode=args.eval_only)
        return model

    def create_model(args):
        if args.arch == 'resnet3D18':
            import models.video_resnet as models
            positional_flag = 1
            model = models.r3d_18(num_classes=args.num_class, pretrained=args.pretrained, spatio_temporal = args.spatio_temporal,positional_flag = positional_flag, num_frames=args.no_frames)
        elif args.arch == 'i3d':
            import models.i3d as models
            model = models.i3d(num_classes=args.num_class, use_gru=args.use_gru, pretrained=args.pretrained,
                               pretrained_path='./models/rgb_imagenet.pt')
        elif args.arch == 'transformer':
            model = init_transformer(args)
        elif args.arch == 'perceiver':
            from models.perceiver_pytorch.perceiver_pytorch import Perceiver

            model = Perceiver(input_channels=3, input_axis=3, depth=3, num_freq_bands=6, max_freq=10, num_classes = args.num_class)

        return model

    def init_contrastive(args):
        # only supporting resnet3d, TODO: Add support for i3d
        def _init_backbone(num_classes):
            from models import video_resnet
            model = video_resnet.r3d_18(num_classes=num_classes, pretrained=args.pretrained, spatio_temporal = args.spatio_temporal, positional_flag=1, num_frames=args.no_frames)
            return model

        if args.arch == 'contrastive':
            from models import contrastive_model
            if args.contrastive_mod == "local-global":
                model = contrastive_model.ContrastiveDecoderModel(_init_backbone, repr_size=args.feature_size)
            elif args.contrastive_mod == "view":
                model = contrastive_model.ContrastiveDecoderModelWithViewClassification(_init_backbone, repr_size=args.feature_size)
            elif args.contrastive_mod == "break-view":
                model = contrastive_model.ContrastiveDecoderModelWithViewClassification(_init_backbone, repr_size=args.feature_size, break_embedding=True)
            elif args.contrastive_mod == "dere":
                model = contrastive_model.ContrastiveDeRecompositionModel(_init_backbone, repr_size=args.feature_size)
            else:
                model = contrastive_model.ContrastiveModel(_init_backbone, repr_size=args.feature_size)

        return model

    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.seed != -1:
        set_seed(args)

    os.makedirs(out_dir, exist_ok=True)

    # remove previous logs
    if args.eval_only or args.no_delete:
        if args.eval_only:
            args.no_clips = 4
    else:
        os.system(f'rm {out_dir}/events.*')
    writer = SummaryWriter(out_dir)



    train_datasets, test_dataset = DATASET_GETTERS[args.dataset]('Data', args.frames_path, num_clips=args.no_clips,
                                                                cross_subject=args.cross_subject, augment=args.augment, contrastive=args.joint_only_multiview_training, args = args, num_frames = args.no_frames, normal_mean=args.mean, normal_std=args.std)

    model = create_model(args) if args.arch != 'contrastive' else init_contrastive(args)
    if args.arch == 'contrastive':
        ### load weights
        if args.eval_only:
            assert os.path.isfile(
                args.resume), "Error: no checkpoint directory found!"

            checkpoint = torch.load(args.resume)
            model.eval_finetune(finetune=args.finetune, endpoint=args.endpoint, num_classes=args.num_class)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            args.pretrained_path = args.resume
            assert os.path.isfile(
                args.pretrained_path), "Error: no checkpoint directory found!"

            checkpoint = torch.load(args.pretrained_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval_finetune(finetune=args.finetune, endpoint=args.endpoint, num_classes=args.num_class)



    model = model.to(args.device)

    args.iteration = sum([len(dataset) for dataset in train_datasets]) // args.batch_size // args.world_size
    train_sampler = RandomSampler
    #train_datasets = [random_split(train_dataset, (round(args.percentage*len(train_dataset)), round((1-args.percentage)*len(train_dataset))))[0] for train_dataset in train_datasets]
    train_loaders = [DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
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

    if args.resume and args.arch != 'contrastive':
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        out_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = start_epoch

    test_accs = []
    model.zero_grad()
    best_loss = math.inf

    # TODO remove later
    decoder=None
    if args.decoder:
        decoder = get_decoder(simple=args.simple, args=args).cuda(1)
        model.fc = None

    if args.eval_only:
        test_loss, test_acc, _, results = test(args, test_loader, model, eval_mode=True)
        os.makedirs('classification', exist_ok=True)
        with open(f'classification/{EXP_NAME}.pickle', 'wb') as f:
            pickle.dump(results, f)

        print("Loss:", test_loss, "Acc:", test_acc)
    else:
        for epoch in range(args.start_epoch, args.epochs):
            train_loss, train_acc = train(args, train_loaders, model, optimizer, scheduler, epoch, decoder=decoder)
            test_loss, test_acc, _, _ = test(args, test_loader, model, eval_mode=False, decoder=decoder)

            '''
            if epoch > (args.epochs+1)/2 and epoch%30==0: 
                test_loss, test_acc, test_acc_2 = test(args, test_loader, test_model, epoch)
            elif epoch == (args.epochs-1):
                test_loss, test_acc, test_acc_2 = test(args, test_loader, test_model, epoch)
            '''

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)

            if decoder is None:
                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)
            else:
                is_best = test_loss < best_loss
                best_loss = min(test_loss, best_loss)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                # TODO remove decoder stuff
                model_to_save = model.module if hasattr(model, "module") else model
                state_dict_to_save = model_to_save.state_dict()
                if decoder:
                    decoder_to_save = decoder.module if hasattr(decoder, "module") else decoder
                    state_dict_to_save = [model_to_save.state_dict(), decoder_to_save.state_dict()]

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': state_dict_to_save,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, out_dir)

            test_accs.append(test_acc)
        with open(f'results/{EXP_NAME}/score_logger.txt', 'a+') as ofile:
            ofile.write(
                f'Last Acc: {test_acc}, Best Acc: {best_acc}\n')

        if args.local_rank in [-1, 0]:
            writer.close()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, indices = torch.topk(output, maxk, dim=1, largest=True, sorted=True)

    res = []
    for k in topk:
        count = 0.0
        for i, t in enumerate(target):
            if t in indices[i, :k]:
                count += 100.0 / batch_size
        res.append(count)
    return res

def calculate_reconstruction_loss(y_hat, x):
    return F.binary_cross_entropy(y_hat.view((y_hat.shape[0], -1)), x.view((x.shape[0], -1)), reduction='none').sum(dim=1).mean()

def get_shuffled_frames(batch_clips, strategy='default'):
    shuffled_clips = batch_clips.detach()
    if strategy == 'default':
        # (batch, clip, channels, frames, h, w)
        shuffled_clips = shuffled_clips[:, :, :, torch.randperm(shuffled_clips.size()[3])]
    elif strategy == 'preserve_first_last':
        shuffled_clips = shuffled_clips[:, :, :, torch.cat((torch.Tensor([0]), torch.randperm(shuffled_clips.size()[3]-2)+1, torch.Tensor([shuffled_clips.size()[3]-1]))).type(torch.int8)]
    elif strategy == 'preserve_first2_last2':
        shuffled_clips = shuffled_clips[:, :, :, torch.cat((torch.Tensor([0,1]), torch.randperm(shuffled_clips.size()[3]-4)+2, torch.Tensor([shuffled_clips.size()[3]-2, shuffled_clips.size()[3]-1]))).type(torch.int8)]

    return shuffled_clips

def train(args, train_loaders, model, optimizer, scheduler, epoch, decoder=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    model.train()

    for train_loader in train_loaders:
        for batch_idx, (inputs_x, targets_x) in enumerate(train_loader):

            data_time.update(time.time() - end)
            inputs = inputs_x.to(args.device)
            targets_x = targets_x.to(args.device)

            logits_x = model(inputs)
            if decoder:
                repr, generated_output = decoder(logits_x.cuda(1))
                if args.sigmoid_loss:
                    loss = calculate_reconstruction_loss(generated_output, inputs.cuda(1))
                else:
                    loss = torch.nn.MSELoss()(generated_output, inputs.cuda(1))
                if args.local_contrastive_loss:
                    logits_x_2 = torch.split(logits_x, logits_x.shape[0]/2)
                    logits_x_2=torch.stack(logits_x_2[1], logits_x_2[0])

                    logits_x = einops.reduce(logits_x, 'b c logits -> b logits', 'mean',
                                             c=args.no_clips)
                    # TODO continue fwd pass
                    loss += 0.3*F.triplet_margin_loss(logits_x, logits_x_2, repr)+0.7*F.cross_entropy(logits_x, targets_x, reduction='mean')
            else:
                if len(logits_x.shape) > 2:
                    logits_x = einops.reduce(logits_x, 'b c logits -> b logits', 'mean',
                                            c=args.no_clips)
                loss = F.cross_entropy(logits_x, targets_x, reduction='mean')
                prec1, prec5 = accuracy(logits_x.data, targets_x, topk=(1, 5))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5, inputs.size(0))

            losses.update(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Top 1 Acc: {acc:.3f}".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.iteration,
                        lr=scheduler.get_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        acc=top1.avg))
                p_bar.update()
    if not args.no_progress:
        p_bar.close()

    return losses.avg, top1.avg


def test(args, test_loader, model, eval_mode=False, decoder=None):
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
                if args.shuffle:
                    inputs = get_shuffled_frames(inputs)
                inputs=einops.rearrange(inputs, 'b c ... -> (b c) ...', c = args.no_clips)
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)

            targets = targets.to(args.device)

            outputs = model(inputs)

            if decoder:
                generated_output = decoder(outputs.cuda(1))
                if args.sigmoid_loss:
                    loss = calculate_reconstruction_loss(generated_output, inputs.cuda(1))
                else:
                    loss = torch.nn.MSELoss()(generated_output, inputs.cuda(1))
            else:

                if eval_mode:
                    outputs = outputs.squeeze()
                    outputs = einops.reduce(outputs, '(b c) logits -> b logits', 'mean',
                                            c=args.no_clips)
                else:
                    if len(outputs.shape) == 3:
                        outputs = einops.reduce(outputs, 'b c logits -> b logits', 'mean',
                                                c=args.no_clips)
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
    main_training_testing()
