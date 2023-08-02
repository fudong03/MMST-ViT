import argparse
import datetime
import json
import math
import sys

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory
from tqdm import tqdm

import util.misc as misc
from dataset.hrrr_loader import HRRR_Dataset
from dataset.sentinel_loader import Sentinel_Dataset
from loss.contrastive_loss import ContrastiveLoss
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models_pvt_simclr import PVTSimCLR
from typing import Iterable
import util.lr_sched as lr_sched
from dataset import sentinel_wrapper


def get_args_parser():
    parser = argparse.ArgumentParser('PVT SimCLR pre-training', add_help=False)

    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--embed_dim', default=512, type=int, help='embed dimensions')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--model', default='pvt_tiny', type=str, metavar='MODEL',
                        help='Name of backbone model to train')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer decay (default: 0.75)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./output_dir/pvt_simclr',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/pvt_simclr',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # dataset
    parser.add_argument('-dr', '--root_dir', type=str, default='/mnt/data/Tiny CropNet')
    parser.add_argument('-tf', '--data_file', type=str, default='./data/soybean_train.json')
    parser.add_argument('-sf', '--save_freq', type=int, default=5)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_sentinel = Sentinel_Dataset(args.root_dir, args.data_file)
    dataset_hrrr = HRRR_Dataset(args.root_dir, args.data_file)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_sentinel = torch.utils.data.DistributedSampler(
            dataset_sentinel, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        sampler_hrrr = torch.utils.data.DistributedSampler(
            dataset_hrrr, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_sentinel = %s" % str(sampler_sentinel))
        print("Sampler_hrrr = %s" % str(sampler_hrrr))
    else:
        sampler_sentinel = torch.utils.data.RandomSampler(dataset_sentinel)
        sampler_hrrr = torch.utils.data.RandomSampler(dataset_hrrr)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_sentinel = torch.utils.data.DataLoader(
        dataset_sentinel, sampler=sampler_sentinel,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=True,
    )

    data_loader_hrrr = torch.utils.data.DataLoader(
        dataset_hrrr, sampler=sampler_hrrr,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=True,
    )

    model = PVTSimCLR(args.model, out_dim=args.embed_dim, context_dim=9, pretrained=True)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_sentinel.sampler.set_epoch(epoch)
            data_loader_hrrr.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_sentinel, data_loader_hrrr,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module,
                    data_loader_sentinel: Iterable, data_loader_hrrr: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    batch_size = args.batch_size

    accum_iter = args.accum_iter
    # criterion = ContrastiveLoss(batch_size, device)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    total_step = len(data_loader_sentinel) - 1
    for data_iter_step, (x, y) in enumerate(zip(data_loader_sentinel, data_loader_hrrr)):

        fips, max_mem = x[1][0], torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        num_grids = tuple(x[0].shape)[2]
        print("Epoch: [{}]  [ {} / {}]  FIPS Code: {}  Number of Grids: {}  Max Mem: {}"
              .format(epoch, data_iter_step, total_step, fips, num_grids, f"{max_mem:.0f}"))

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_sentinel) + epoch, args)

        # prevent the number of grids from being too large to cause out of memory
        train_loader = sentinel_wrapper.get_data_loader(x[0], y[0], batch_size=batch_size)

        for xi, xj, ys in train_loader:
            xi = xi.to(device, non_blocking=True)
            xj = xj.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            zi = model(xi, ys)
            zj = model(xj, ys)

            criterion = ContrastiveLoss(zi.shape[0], device)
            loss = criterion(zi, zj)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader_sentinel) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
