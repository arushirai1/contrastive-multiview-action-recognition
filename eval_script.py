import argparse
import math
import os
import shutil
import time

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
NTUARD_SUPERVISED_TRAININGi3d1664
NTUARD_SUPERVISED_TRAININGi3d1664_True
NTUARD_SUPERVISED_TRAININGresnet3D181664
NTUARD_SUPERVISED_TRAININGresnet3D181664_True
'''
def save_checkpoint(state, is_best, checkpoint):
    filename = f'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'model_best.pth.tar'))


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
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate():
    parser = argparse.ArgumentParser(description='PyTorch Evaluation Script')
    parser.add_argument('--modeldir', default='', help='directory to model')
    parser.add_argument('--arch', default='i3d', help='Model architecture')
    parser.add_argument('--num-class', default=60, type=int,
                        help='total classes')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--dataset', default='ucf101', type=str,
                        help='dataset name')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    args = parser.parse_args()

    def create_model(args):
        if args.arch == 'resnet3D18':
            import models.video_resnet as models
            model = models.r3d_18(num_classes=args.num_class)
        elif args.arch == 'i3d':
            import models.i3d as models
            model = models.i3d(num_classes=args.num_class)
        return model

    args.device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()

    _, test_dataset = DATASET_GETTERS[args.dataset]('Data', args.frames_path)

    model = create_model(args)
    model.to(args.device)


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
    model.load_state_dict(torch.load(os.path.join(args.modelpath, 'model_best.pth.tar')))

    test_loss, test_top1, test_top5 = test(args, test_loader, model)
    print(test_loss, test_top1, test_top5)

def test(args, test_loader, model, ):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            targets = targets.cpu().numpy().tolist()
            outputs = outputs.cpu().numpy().tolist()

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            test_loader.set_description(
                "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Top 1 Acc: {acc:.3f}".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    acc=top1.avg
                ))
        test_loader.close()
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    cudnn.benchmark = True
    evaluate()
