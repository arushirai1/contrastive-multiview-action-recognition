import argparse
import math
import os
import shutil
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from Data.UCF101 import get_ucf101, get_ntuard
from utils import AverageMeter, accuracy

DATASET_GETTERS = {'ucf101': get_ucf101, 'ntuard': get_ntuard}

'''
results/NTUARD_SUPERVISED_TRAININGi3d1664/score_logger.txt
results/NTUARD_SUPERVISED_TRAININGi3d1664_True/score_logger.txt
results/NTUARD_SUPERVISED_TRAININGresnet3D181664/score_logger.txt
results/NTUARD_SUPERVISED_TRAININGresnet3D181664_True/score_logger.txt

NTUARD_SUPERVISED_TRAININGi3d1664
NTUARD_SUPERVISED_TRAININGi3d1664_True
NTUARD_SUPERVISED_TRAININGresnet3D181664
NTUARD_SUPERVISED_TRAININGresnet3D181664_True

1/21/2021
results/NTUARD_SUPERVISED_TRAININGi3d832_False_clips_3/
results/NTUARD_SUPERVISED_TRAININGi3d832_True_clips_3/

NTUARD_SUPERVISED_TRAININGi3d832_False_clips_3
NTUARD_SUPERVISED_TRAININGi3d832_True_clips_3
NTUARD_SUPERVISED_TRAININGresnet3D181616_False_clips_3
NTUARD_SUPERVISED_TRAININGresnet3D181616_True_clips_3

NTUARD_SUPERVISED_TRAININGresnet3D181616_True_clips_3_gru_False
NTUARD_SUPERVISED_TRAININGi3d1664_True_clips_3
'''
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, indices = torch.topk(output, maxk, dim=1, largest=True, sorted=True)

    res = []
    for k in topk:
        count=0.0
        for i, t in enumerate(target):
            if t in indices[i,:k]:
                count+=100.0 / batch_size
        res.append(count)
    return res

def evaluate():
    parser = argparse.ArgumentParser(description='PyTorch Evaluation Script')
    parser.add_argument('--modeldir', default='', help='directory to model')
    parser.add_argument('--arch', default='i3d', help='Model architecture')
    parser.add_argument('--num-class', default=60, type=int,
                        help='total classes')
    parser.add_argument('--frames-path', default='/datasets/UCF-101/Frames/frames-128x128/', type=str,
                        help='video frames path')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--no-clips', default=1, type=int,
                        help='number of clips')
    parser.add_argument('--dataset', default='ucf101', type=str,
                        help='dataset name')
    parser.add_argument('--use-gru', action='store_true', default=False,
                        help='use gru')
    parser.add_argument('--cross-subject', action='store_true', default=False,
                        help='Training and testing on cross subject split')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    args = parser.parse_args()

    def create_model(args):
        if args.arch == 'resnet3D18':
            import models.video_resnet as models
            model = models.r3d_18(num_classes=args.num_class)
        elif args.arch == 'i3d':
            import models.i3d as models
            model = models.i3d(num_classes=args.num_class, use_gru=args.use_gru)
        return model

    args.device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()

    _, test_dataset = DATASET_GETTERS[args.dataset]('Data', args.frames_path, num_clips=args.no_clips, cross_subject=args.cross_subject)

    model = create_model(args)


    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)
    loaded_checkpoint=torch.load(os.path.join(args.modeldir, 'model_best.pth.tar'))
    state_dict = loaded_checkpoint['state_dict']
    print(loaded_checkpoint['best_acc'])
    exit(0)
    model.load_state_dict(state_dict)
    model.to(args.device)
    test_loss, test_top1, test_top5 = test(args, test_loader, model)
    print(test_loss, test_top1, test_top5)

    '''

    test_loss, test_top1, test_top5, predictions_by_ground_truth = test2(args, test_loader, model)
    print(test_loss, test_top1, test_top5)
    with open('prediction_by_ground_truth.pickle', 'wb') as f:
        import pickle
        pickle.dump(predictions_by_ground_truth, f)
    with open('ucf101-supervised/prediction_by_ground_truth.pickle', 'rb') as f:
        import pickle
        predictions_by_ground_truth=pickle.load(f)
    list(list(zip(*predictions_by_ground_truth.values()))[1])
    indexes = np.argpartition(np.array(list(list(zip(*predictions_by_ground_truth.values()))[1])), -5)[-5:]
    np.array(list(predictions_by_ground_truth.keys()))[indexes]
    '''

def test2(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    predicted_target = {}

    if True:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            loss = F.cross_entropy(outputs, targets)
            targets = targets.cpu().numpy().tolist()
            outputs = outputs.cpu().numpy().tolist()

            for iterator in range(len(targets)):
                # init key,val in dicts if empty
                if targets[iterator] not in predicted_target:
                    predicted_target[targets[iterator]] = []

                predicted_target[targets[iterator]].append(outputs[iterator])


            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if True:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Top 1 Acc: {acc:.3f}".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        acc=top1.avg
                    ))
        if True:
            test_loader.close()

    for key in predicted_target:
        clip_values = np.array(predicted_target[key])

        video_pred = np.argmax(clip_values, axis=1)
        acc_by_key = (video_pred.shape[0]-np.count_nonzero(video_pred-key))/video_pred.shape[0]
        predicted_target[key] = (video_pred, acc_by_key)
    return losses.avg, top1.avg, top5.avg, predicted_target


def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    if not False:
        test_loader = tqdm(test_loader)

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction='mean')

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not False:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Top 1 Acc: {acc:.3f}".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        acc=top1.avg
                    ))
        if not False:
            test_loader.close()

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    cudnn.benchmark = True
    evaluate()
