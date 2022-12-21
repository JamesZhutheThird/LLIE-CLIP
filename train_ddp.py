import os
import pdb
import random
import sys
import time
import glob
import numpy as np
import pandas as pd
import torch
import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from model import *
from multi_read_data import MemoryFriendlyLoader
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.optim.lr_scheduler import StepLR
import wandb

from metrics import get_metrics

def set_random_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    parser = argparse.ArgumentParser("SCI")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--global_rank", type=int, default=-1)
    parser.add_argument("--eval", action='store_true', help="eval model")
    parser.add_argument("--eval_per_epoch", type=int, default=1, help="perform evaluation per epoch")
    parser.add_argument("--model_save_per_epoch", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument("--schedular_step", type=int, default=1, help="step size for schedular")
    parser.add_argument('--stage', type=int, default=3, help='epochs')
    parser.add_argument('--train_dir', type=str,
                        help='path to train data')
    parser.add_argument('--test_dir', type=str,
                        help='path to test data')
    parser.add_argument('--output_dir', type=str, help='path to save results')
    parser.add_argument('--model', type=str, default=None, help='pretrained model path')
    parser.add_argument("--exp_name", type=str, default='llie-sci', help="name of experiment")
    parser.add_argument("--wandb", action='store_true', help="enable wandb")
    parser.add_argument("--wandb_project_name", type=str, default="LLIE-SCI", help="name of wandb project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="name of wandb run")
    parser.add_argument("--start_time", type=str, default=None, help="start time of experiment")
    parser.add_argument("--inception", action='store_true', help="use inception networks")
    
    args = parser.parse_args()
    
    if args.global_rank >= 0:
        args.local_rank = int(os.environ.get("LOCAL_RANK"))
    
    save_path = f"{args.output_dir}/{args.exp_name}/{args.start_time}"
    model_path = f"{args.output_dir}/{args.exp_name}/{args.start_time}/models"
    image_path = f"{args.output_dir}/{args.exp_name}/{args.start_time}/images"
    log_path = f"{args.output_dir}/{args.exp_name}/{args.start_time}/logs"
    
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(f"{log_path}/logs.txt")
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        
        logging.info("train file name = %s", os.path.split(__file__))
    
    if args.wandb_run_name is None:
        args.wandb_run_name = args.exp_name
    
    set_random_seed(args)
    
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.total_world_size = args.n_gpu
        args.total_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps * args.total_world_size
        if args.n_gpu == 0:
            print("No GPU available. Training on CPU.")
        elif args.n_gpu == 1:
            print("Only 1 GPU available. Training on GPU.")
        else:
            print(f"{args.n_gpu} GPUs available. Training with DistributedParallel.")
    else:
        print("Loading GPU {}.".format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        args.n_gpu = 1
        args.total_world_size = int(os.environ.get("WORLD_SIZE"))  # torch.distributed.get_world_size()
        if args.local_rank == 0:
            print(f"{args.total_world_size} GPUs available. Training with DistributedDataParallel.")
        local_id = args.local_rank
        torch.distributed.init_process_group("nccl")
        device = torch.device("cuda", local_id)
        print(device)
    args.device = device
    
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    model = Network(stage=args.stage,inception=args.inception)
    
    if args.model is not None:
        model.load_state_dict(torch.load(args.model, map_location=device))
        if args.local_rank in [-1, 0]:
            print("load model from %s", args.model)
    else:
        model.enhance.in_conv.apply(model.weights_init)
        model.enhance.conv.apply(model.weights_init)
        model.enhance.out_conv.apply(model.weights_init)
        model.calibrate.in_conv.apply(model.weights_init)
        model.calibrate.convs.apply(model.weights_init)
        model.calibrate.out_conv.apply(model.weights_init)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    MB = utils.count_parameters_in_MB(model)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    if args.local_rank in [-1, 0]:
        print(f"model size = {MB}, parameter number = {para}")
    
    if args.local_rank == 0:
        torch.distributed.barrier()
    
    if args.local_rank in [-1, 0] and args.wandb:
        # save wandb project in team account jameszhuthethirdteam
        wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, dir=log_path, entity="jameszhuthethirdteam")
        wandb.config = {"args": args, }
        wandb.define_metric("epoch")
        wandb.define_metric("loss", step_metric='epoch')
        wandb.define_metric("lr", step_metric='epoch')
    
    model.to(args.device)
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif args.n_gpu > 1:
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
    
    args.total_batch_size = args.batch_size * max(1, args.n_gpu)
    
    if args.local_rank in [-1, 0]:
        logging.info("args = %s", args)
    
    total_step = 0
    
    with tqdm(range(args.epochs), disable=args.local_rank not in [-1, 0]) as t1:
        for epoch in t1:
            t1.set_description(f"Epoch {epoch}")
            
            TrainDataset = MemoryFriendlyLoader(img_dir=args.train_dir, task='train')
            train_sampler = RandomSampler(TrainDataset) if args.local_rank == -1 else DistributedSampler(TrainDataset)
            train_queue = DataLoader(TrainDataset, sampler=train_sampler, batch_size=args.total_batch_size)
            
            model.train()
            losses = []
            with tqdm(train_queue, disable=args.local_rank not in [-1, 0]) as t2:
                for batch_idx, (input, _) in enumerate(t2):
                    t2.set_description(f"Train on Epoch {epoch}")
                    input = Variable(input, requires_grad=False).cuda()
                    optimizer.zero_grad()
                    loss = model.module._loss(input)
                    if args.n_gpu > 1:
                        loss = loss.mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    optimizer.zero_grad()
                    losses.append(loss.item())
                    t2.set_postfix(loss=loss.item())
                    if args.local_rank in [-1, 0] and args.wandb:
                        wandb.log({"loss": loss, "lr": scheduler.get_last_lr()[0], 'epoch': epoch + batch_idx / len(train_queue)})
                    total_step += 1
                    
                    # # Would you add it to debug mode please =v=
                    # if batch_idx == 2:
                    #     break
                    
            if args.local_rank in [-1, 0]:
                logging.info('train-epoch %03d %f', epoch, np.average(losses))
            
            if (epoch + 1) % args.schedular_step == 0:
                scheduler.step()
            
            if args.local_rank not in [-1, 0]:
                torch.distributed.barrier()
            
            if args.local_rank in [-1, 0]:
                
                if (epoch + 1) % args.model_save_per_epoch == 0:
                    torch.save(model.state_dict(), f"{model_path}/model_{epoch + 1}.pth")
                    logging.info("save model to %s", f"{model_path}/model_{epoch + 1}.pth")
                 
                if args.eval and (epoch + 1) % args.eval_per_epoch == 0 and total_step != 0:
                    TestDataset = MemoryFriendlyLoader(img_dir=args.test_dir, task='test')
                    eval_sampler = SequentialSampler(TestDataset)
                    test_queue = DataLoader(TestDataset, sampler=eval_sampler, batch_size=args.n_gpu)

                    output_path_for_eval = f"{image_path}/{epoch + 1}"
                    os.makedirs(output_path_for_eval,exist_ok=True)
                    
                    model.eval()
                    torch.cuda.empty_cache()
                    
                    with torch.no_grad():
                        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                            model = torch.nn.DataParallel(model)
                        
                        metrics_df = None
                        
                        with tqdm(test_queue) as t3:
                            for _, (input, image_name) in enumerate(t3):
                                # Argument volatile was removed and now has no effect. Statement `with torch.no_grad():` has been used instead.
                                input = Variable(input).cuda()
                                # Using '/' as seperator (Linux), origin repo runs on Windows
                                image_name = image_name[0].split('/')[-1].split('.')[0]
                                illu_list, ref_list, input_list, atten = model(input)
                                u_path = f"{output_path_for_eval}/{image_name}_{epoch + 1}.png"
                                save_images(ref_list[0], u_path)
                                
                                metrics_batch = get_metrics(image_name, ref_list[0], input)
                                if metrics_df is not None:
                                    metrics_df = pd.concat([metrics_df, metrics_batch],axis=0)
                                else:
                                    metrics_df = metrics_batch
                    
                        metrics_df.to_csv(f"{output_path_for_eval}/metrics_{epoch + 1}.csv")
                        logging.info("evaluation-epoch %03d",epoch + 1)
                        
                        for metics_name in metrics_df.columns:
                            logging.info("metric %s = %f", metics_name, metrics_df[metics_name].mean())
                            if args.wandb:
                                wandb.log({f"metric_{metics_name}": metrics_df[metics_name].mean()})
                        
                        # pdb.set_trace()
            
            if args.local_rank == 0:
                torch.distributed.barrier()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
