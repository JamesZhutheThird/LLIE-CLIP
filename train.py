import os
import random
import sys
import time
import glob
import numpy as np
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
    parser.add_argument("--eval", action='store_true', help="eval model")
    parser.add_argument("--eval_per_epoch", type=int, default=1, help="perform evaluation per epoch")
    parser.add_argument("--model_save_per_epoch", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
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
    
    args = parser.parse_args()
    
    args.start_time = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"{args.output_dir}/{args.exp_name}/{args.start_time}"
    model_path = f"{args.output_dir}/{args.exp_name}/{args.start_time}/models"
    image_path = f"{args.output_dir}/{args.exp_name}/{args.start_time}/images"
    log_path = f"{args.output_dir}/{args.exp_name}/{args.start_time}/logs"
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    set_random_seed(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(f"{log_path}/logs.txt")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    logging.info("train file name = %s", os.path.split(__file__))
    
    if args.wandb_run_name is None:
        args.wandb_run_name = args.exp_name
    
    logging.info("args = %s", args)
    
    model = Network(stage=args.stage)
    
    if args.model is not None:
        model.load_state_dict(torch.load(args.model))
        logging.info("load model from %s", args.model)
    else:
        model.enhance.in_conv.apply(model.weights_init)
        model.enhance.conv.apply(model.weights_init)
        model.enhance.out_conv.apply(model.weights_init)
        model.calibrate.in_conv.apply(model.weights_init)
        model.calibrate.convs.apply(model.weights_init)
        model.calibrate.out_conv.apply(model.weights_init)
    
    model.to(args.device)  # TODO: add DistributedDataParallel support
    
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, dir=log_path)
        wandb.config = {"args": args, }
        wandb.define_metric("epoch")
        wandb.define_metric("loss", step_metric='epoch')
        wandb.define_metric("lr", step_metric='epoch')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    logging.info("parameter number = %f", para)
    print(f"model size = {MB}, parameter number = {para}")
    
    TrainDataset = MemoryFriendlyLoader(img_dir=args.train_dir, task='train')
    
    TestDataset = MemoryFriendlyLoader(img_dir=args.test_dir, task='test')
    
    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True)
    
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True)
    
    total_step = 0
    
    with tqdm(range(args.epochs)) as t1:
        for epoch in t1:
            t1.set_description(f"Epoch {epoch}")
            model.train()
            losses = []
            with tqdm(train_queue) as t2:
                for batch_idx, (input, _) in enumerate(t2):
                    t2.set_description(f"Train on Epoch {epoch}")
                    input = Variable(input, requires_grad=False).cuda()
                    optimizer.zero_grad()
                    loss = model._loss(input)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    losses.append(loss.item())
                    t2.set_postfix(loss=loss.item())
                    if args.wandb:
                        wandb.log({"loss": loss, "lr": optimizer.param_groups[0]['lr'], 'epoch': epoch + batch_idx / len(train_queue)})
            logging.info('train-epoch %03d %f', epoch, np.average(losses))
            
            if (epoch + 1) % args.model_save_per_epoch == 0:
                torch.save(model.state_dict(), f"{model_path}/model_{epoch + 1}.pth")
                logging.info("save model to %s", f"{model_path}/model_{epoch + 1}.pth")
            
            if args.eval and epoch % args.eval_per_epoch == 0 and total_step != 0:
                model.eval()
                with torch.no_grad():
                    with tqdm(enumerate(test_queue)) as t3:
                        for _, (input, image_name) in t3:
                            input = Variable(input, volatile=True).cuda()
                            image_name = image_name[0].split('\\')[-1].split('.')[0]
                            illu_list, ref_list, input_list, atten = model(input)
                            u_path = f"{image_path}/{image_name}_{epoch + 1}.png"
                            save_images(ref_list[0], u_path)


if __name__ == '__main__':
    main()
