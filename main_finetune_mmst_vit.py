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
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from dataset.data_wrapper import DataWrapper
from dataset.hrrr_loader import HRRR_Dataset
from dataset.sentinel_loader import Sentinel_Dataset
from dataset.usda_loader import USDA_Dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models_pvt_simclr import PVTSimCLR
from typing import Iterable
import util.lr_sched as lr_sched
from models_mmst_vit import MMST_ViT
from util import metrics

from datetime import datetime

torch.manual_seed(0)
np.random.seed(0)

# RMSE, R_Squared, Corr
best_metrics = [float("inf"), 0, 0]


def get_args_parser():
    parser = argparse.ArgumentParser('MMST-ViT fine-tuning', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--embed_dim', default=512, type=int, help='embed dimensions')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--model_pvt', default='pvt_tiny', type=str, metavar='MODEL',
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

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./output_dir/mmst_vit',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/mmst_vit',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

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
    parser.add_argument('-sf', '--save_freq', type=int, default=2)

    # train and val
    parser.add_argument('-dft', '--data_file_train', type=str, default='./data/soybean_train.json')
    parser.add_argument('-dfv', '--data_file_val', type=str, default='./data/soybean_val.json')

    # pvt_simclr
    parser.add_argument('--pvt_simclr', default='', help='load from checkpoint')

    # evaluate
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_year', type=int, default=2022, help='specify the year for prediction')

    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')

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

    dataset_sentinel_train = Sentinel_Dataset(args.root_dir, args.data_file_train)
    dataset_hrrr_train = HRRR_Dataset(args.root_dir, args.data_file_train)
    dataset_usda_train = USDA_Dataset(args.root_dir, args.data_file_train)

    dataset_sentinel_val = Sentinel_Dataset(args.root_dir, args.data_file_val)
    dataset_hrrr_val = HRRR_Dataset(args.root_dir, args.data_file_val)
    dataset_usda_val = USDA_Dataset(args.root_dir, args.data_file_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        # train
        sampler_sentinel_train = torch.utils.data.DistributedSampler(
            dataset_sentinel_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_hrrr_train = torch.utils.data.DistributedSampler(
            dataset_hrrr_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_usda_train = torch.utils.data.DistributedSampler(
            dataset_usda_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        print("Sampler_sentinel_train = %s" % str(sampler_sentinel_train))
        print("Sampler_hrrr_train = %s" % str(sampler_hrrr_train))
        print("Sampler_usda_train = %s" % str(sampler_usda_train))

        # val
        sampler_sentinel_val = torch.utils.data.DistributedSampler(
            dataset_sentinel_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_hrrr_val = torch.utils.data.DistributedSampler(
            dataset_hrrr_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_usda_val = torch.utils.data.DistributedSampler(
            dataset_usda_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        print("Sampler_sentinel_val = %s" % str(sampler_sentinel_val))
        print("Sampler_hrrr_val = %s" % str(sampler_hrrr_val))
        print("Sampler_usda_val = %s" % str(sampler_usda_val))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # train
    data_loader_sentinel_train = torch.utils.data.DataLoader(
        dataset_sentinel_train, sampler=sampler_sentinel_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_hrrr_train = torch.utils.data.DataLoader(
        dataset_hrrr_train, sampler=sampler_hrrr_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_usda_train = torch.utils.data.DataLoader(
        dataset_usda_train, sampler=sampler_usda_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # val
    data_loader_sentinel_val = torch.utils.data.DataLoader(
        dataset_sentinel_val, sampler=sampler_sentinel_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_hrrr_val = torch.utils.data.DataLoader(
        dataset_hrrr_val, sampler=sampler_hrrr_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_usda_val = torch.utils.data.DataLoader(
        dataset_usda_val, sampler=sampler_usda_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    pvt = PVTSimCLR(args.model_pvt, out_dim=args.embed_dim, context_dim=9)
    if args.pvt_simclr:
        checkpoint = torch.load(args.pvt_simclr, map_location='cpu')
        pvt.load_state_dict(checkpoint['model'])
        pvt.to(device)
        pvt.eval()

    model = MMST_ViT(out_dim=2, pvt_backbone=pvt, context_dim=9, dim=args.embed_dim, batch_size=args.batch_size)
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

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        evaluate(model, data_loader_sentinel_train, data_loader_hrrr_train, data_loader_usda_train, device)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # train
            data_loader_sentinel_train.sampler.set_epoch(epoch)
            data_loader_hrrr_train.sampler.set_epoch(epoch)
            data_loader_usda_train.sampler.set_epoch(epoch)

            # val
            data_loader_sentinel_val.sampler.set_epoch(epoch)
            data_loader_hrrr_val.sampler.set_epoch(epoch)
            data_loader_usda_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_sentinel_train, data_loader_hrrr_train, data_loader_usda_train,
            optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            # evaluate
            evaluate(model, data_loader_sentinel_val, data_loader_hrrr_val, data_loader_usda_val, device)

            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module,
                    data_loader_sentinel: Iterable, data_loader_hrrr: Iterable, data_loader_usda: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter
    # data augmentation by following SimCLR
    data_wrapper = DataWrapper()

    criterion = torch.nn.MSELoss().to(device)
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    total_step = len(data_loader_sentinel) - 1
    for data_iter_step, (x, y, z) in enumerate(zip(data_loader_sentinel, data_loader_hrrr, data_loader_usda)):

        fips, max_mem = x[1][0], torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        num_grids = tuple(x[0].shape)[2]
        print("Epoch: [{}]  [ {} / {}]  FIPS Code: {}  Number of Grids: {}  Max Mem: {}"
              .format(epoch, data_iter_step, total_step, fips, num_grids, f"{max_mem:.0f}"))

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_sentinel) + epoch, args)


        # satellited imagery
        x = x[0].to(device, non_blocking=True)
        # short- and long-term weather variables
        ys = y[0].to(device, non_blocking=True)
        yl = y[1].to(device, non_blocking=True)
        # USDA
        z = z[0].to(device, non_blocking=True)

        b, t, g, _, _, _ = x.shape
        x = rearrange(x, 'b t g h w c -> (b t g) c h w')
        x, _ = data_wrapper(x)

        x = rearrange(x, '(b t g) c h w -> b t g c h w', b=b, t=t, g=g)


        z_hat = model(x, ys=ys, yl=yl)

        # log_scale
        loss = criterion(z, z_hat)
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


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader_sentinel: Iterable, data_loader_hrrr: Iterable,
             data_loader_usda: Iterable, device: torch.device):
    # data augmentation by following SimCLR
    data_wrapper = DataWrapper(train=False)

    true_labels = torch.empty(0)
    pred_labels = torch.empty(0)

    total_step = len(data_loader_sentinel) - 1
    for data_iter_step, (x, y, z) in enumerate(zip(data_loader_sentinel, data_loader_hrrr, data_loader_usda)):

        fips, max_mem = x[1][0], torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        num_grids = tuple(x[0].shape)[2]
        print(" Eval [ {} / {}]  FIPS Code: {}  Number of Grids: {}  Max Mem: {}"
              .format(data_iter_step, total_step, fips, num_grids, f"{max_mem:.0f}"))

        # satellite imagery
        x = x[0].to(device, non_blocking=True)

        # short- and long-term weather variables
        ys = y[0].to(device, non_blocking=True)
        yl = y[1].to(device, non_blocking=True)

        # USDA
        z = z[0].to(device, non_blocking=True)

        b, t, g, _, _, _ = x.shape
        x = rearrange(x, 'b t g h w c -> (b t g) c h w')
        x, _ = data_wrapper(x)
        x = rearrange(x, '(b t g) c h w -> b t g c h w', b=b, t=t, g=g)

        z_hat = model(x, ys=ys, yl=yl)

        true_labels = torch.cat([true_labels, z.detach().cpu()], dim=0)
        pred_labels = torch.cat([pred_labels, z_hat.detach().cpu()], dim=0)


    true_labels = torch.exp(torch.flatten(true_labels[:, -1:], start_dim=0)).detach().cpu().numpy()
    pred_labels = torch.exp(torch.flatten(pred_labels[:, -1:], start_dim=0)).detach().cpu().numpy()

    rmse, r2, corr = metrics.evaluate(true_labels, pred_labels)

    global best_metrics
    best_metrics = [min(best_metrics[0], rmse), max(best_metrics[1], r2), max(best_metrics[2], corr)]
    print("Metrics: RMSE: {}  R_Squared: {}  Corr: {}".format(f"{best_metrics[0]:.2f}", f"{ best_metrics[1]:.2f}", f"{best_metrics[2]:.2f}"))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
