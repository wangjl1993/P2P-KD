'''
Descripttion: 
version: 
Author: wangjl
Date: 2021-03-22 14:47:47
LastEditors: pystar360 pystar360@py-star.com
LastEditTime: 2024-01-31 09:10:55
'''

import torch

import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

from pathlib import Path
import numpy as np
import random
import os
import argparse
from copy import deepcopy
from tqdm import tqdm
import json

import incremental_dataloader as build_data
from utils import (
    InputNormalize, make_step, 
    random_perturb, Logger,
    get_linea_lr, get_sgdr_lr,
    compute_acc_fgt
)
from alexnet import AlexNet
from resnet_cifar import ResNet18

parser = argparse.ArgumentParser(description='PyTorch P2P-KD for domainIL')
parser.add_argument('--data_path', default='/media/datum/wangjl/data/iTAML-imagenet/224/animal10', type=str, help='data path')
parser.add_argument('--class_per_task', default=10, type=int)
parser.add_argument('--num_class', default=40, type=int, help='all classes')
parser.add_argument('--num_task', default=4, type=int)
parser.add_argument('--dataset', default='animal_imagenet', type=str)
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--optim', default='sgdm', type=str, choices=['sgdm', 'adam'], help='optimizer')
parser.add_argument('--train_batch', default=64, type=int)
parser.add_argument('--test_batch', default=50, type=int, help='num of per class')
parser.add_argument('--decay', default='linear', type=str, help='lr decay strategy')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--memory_size', type=int, default=0, help='memory size, set it huge for training full data')
parser.add_argument('--mu', type=int, default=1)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--schedule', nargs="+", type=int, default=[-1], help='The list of epoch numbers to reduce learning rate by factor 0.1, -1 didnt decay.')
parser.add_argument('--resume', action='store_true', help='resume first task model')
parser.add_argument('--ckpt', type=str, help='pretrained parameters for resume')
parser.add_argument('--kd', action='store_true', help='knowledge distill')
parser.add_argument('--alpha', default=1.0, type=float, help='kd loss factor')
parser.add_argument('--T', default=2., type=float, help='temperature of kd')
parser.add_argument('--th_p', type=float, default=0.9)
parser.add_argument('--iters', type=int, default=10)
parser.add_argument('--task_order', type=str, default='0123', help='task order')
parser.add_argument('--selected_th_p', type=float, help='if None, select correct index. otherwise select those whose p > selected_th_p ')
parser.add_argument('--method', default="P2P-KD", choices=['oracle', 'P2P-KD', 'baseline'], help='method')
parser.add_argument('--device', default=0, type=int)

args = parser.parse_args()

seed = args.seed
print('seed = ', seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


save_path = Path(f"results/{args.dataset}/{args.task_order}/{args.method}/seed{args.seed}_{args.optim}_batchsize{args.train_batch}_{args.decay}_memorysize{args.memory_size}")
save_path.mkdir(exist_ok=True, parents=True)
w_logger = Logger(save_path)


print(args)
with open(os.path.join(save_path,'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


# 对迁移数据做过滤
def train(epoch):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    train_loss1 = 0
    train_loss2 = 0
    aug_num = 0
    c1, c2, t1, t2 = 0, 0, 0, 0 # 旧模型预测新数据/迁移数据正确的数量
    
    for batch_idx, (data, tar) in enumerate(tqdm(mytrainloader)):
        tar %= args.class_per_task
        data, tar = data.to(device), tar.to(device)

        if args.kd and sess > 0:
            with torch.no_grad():
                old_output = model(normalizer(data))
                c1 += old_output.max(1)[1].eq(tar).sum().item()
                t1 += old_output.shape[0]
            p = old_output.softmax(dim=1)
            tar_p = torch.tensor([p[i,j] for i,j in enumerate(tar)])
            private_idx = tar_p < args.th_p # private domain data index
            
            if private_idx.sum().item() > 0:
                
                aug_data, aug_tar = deepcopy(data[private_idx]), deepcopy(tar[private_idx])

                # for private domain data, ce loss
                output = net(normalizer(aug_data))
                loss1 = criterion(output, aug_tar)
                pre = output.max(1)[1]
                correct += pre.eq(aug_tar).sum().item()
                total += aug_data.shape[0]
                train_loss1 += loss1.item()

                # P2P
                random_noise = random_perturb(aug_data, 'l2', 0.5)
                aug_data = torch.clamp(aug_data+random_noise, min=0, max=1)
                aug_data.requires_grad_(True)
                for _ in range(args.iters):
                    out = model(normalizer(aug_data))
                    img_loss = criterion(out, aug_tar)
                    grad, = torch.autograd.grad(img_loss, [aug_data])

                    aug_data = aug_data - make_step(grad, 'l2', 0.1)
                    aug_data = torch.clamp(aug_data, min=0, max=1)

                # 选取符合的迁移数据
                aug_data.detach_()
                with torch.no_grad():
                    aug_output = model(normalizer(aug_data))
                    if args.selected_th_p is None:
                        aug_pre = aug_output.max(1)[1]
                        selected_aug_idx = aug_pre.eq(aug_tar)
                    else:
                        aug_p = aug_output.softmax(dim=1)
                        aug_tar_p = torch.tensor([aug_p[i,j] for i,j in enumerate(aug_tar)])
                        selected_aug_idx = aug_tar_p > args.selected_th_p 
                aug_num += selected_aug_idx.sum().item()
                # for public data, knowledge distill loss
                
                data = torch.cat([data[~private_idx], aug_data[selected_aug_idx]])
                tar = torch.cat([tar[~private_idx], aug_tar[selected_aug_idx]])
                if len(data) > 0:
                    output1 = net(normalizer(data))   
                    with torch.no_grad():
                        output2 = model(normalizer(data))  
                        c2 += output2.max(1)[1].eq(tar).sum().item()
                        t2 += output2.shape[0]
                        
                    log_p = (output1/args.T).log_softmax(dim=1)
                    q = (output2/args.T).softmax(dim=1)
                    loss2 = F.kl_div(log_p, q, reduction='batchmean')
                    train_loss2 += loss2.item()
                    
                    correct += output1.max(1)[1].eq(tar).sum().item()
                    total += data.shape[0]
                    
                else:
                    loss2 = 0
                
                loss = loss1 + loss2*args.T*args.T*args.alpha

            else:
                output1 = net(normalizer(data))
                with torch.no_grad():   
                    output2 = model(normalizer(data)) 
                
                log_p = (output1/args.T).log_softmax(dim=1)
                q = (output2/args.T).softmax(dim=1)
                loss = F.kl_div(log_p, q, reduction='batchmean')
                train_loss2 += loss.item()

                correct += output1.max(1)[1].eq(tar).sum().item()
                total += data.shape[0]
        else:
            output = net(normalizer(data))
            loss = criterion(output, tar)
            pre = output.max(1)[1]
            total += data.shape[0] 
            correct += pre.eq(tar).sum().item()                       
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
    msg = 'epoch {0:d} | loss {1:.4f}, loss1 {2:.4f}, loss2 {3:.4f} | train_acc {4:.4f} , {5:d}/{6:d} | lr {7:.4f}'.format(
        epoch, 
        train_loss/len(mytrainloader), 
        train_loss1/len(mytrainloader), 
        train_loss2/len(mytrainloader), 
        correct/total,
        correct, 
        total, 
        cur_lr
    )
    w_logger.log(msg)

def test(epoch):
    net.eval()
    
    task_correct = dict.fromkeys(range(args.num_task), 0)
    task_total = dict.fromkeys(range(args.num_task), 1e-8)

    with torch.no_grad():
        for batch_idx, (data, tar) in enumerate(tqdm(mytestloader)):
            which_task = tar // args.class_per_task
            tar = tar % args.class_per_task
            
            data, tar = data.to(device), tar.to(device)
            output = net(normalizer(data))
            pre = output.max(1)[1]
            
            for i in range(args.num_task):
                task_total[i] += (which_task==i).sum().item()
                task_correct[i] += pre[which_task==i].eq(tar[which_task==i]).sum().item()

        task_acc = {k1:v1/v2 for (k1,v1), (k2,v2) in zip(task_correct.items(), task_total.items())}
        Acc = sum(task_correct.values()) / sum(task_total.values())

        msg = [f"acc={Acc:.4f}"]
        msg += [f"task{i}_acc={task_acc[i]:.4f}" for i in range(args.num_task)]
        msg = " | ".join(msg)
        print(msg)
        if epoch < 5:
            print(task_correct, task_total)
        w_logger.log(msg)
    return Acc, task_acc






if __name__ == '__main__':

    device = torch.device(args.device)

    acc_dict = {'task': []}
    for i in range(args.num_task):
        acc_dict[i] = []

    classes_order = []
    for task_id in args.task_order:
        task_id = int(task_id)
        classes_order += range(args.class_per_task*task_id, args.class_per_task*(task_id+1))
    # Data
    inc_dataset = build_data.IncrementalDataset(
        dataset_name=args.dataset,
        root=args.data_path,
        order=classes_order,
        batch_size=args.train_batch,
        workers=16,
    )


    # Model
    if args.dataset == 'animal_imagenet':
        net = models.resnet18(num_classes=args.class_per_task).to(device)
        normalizer = InputNormalize(
            new_mean=torch.tensor([0.485, 0.456, 0.406]),
            new_std=torch.tensor([0.229, 0.224, 0.225]),
            device=device
        )
    elif args.dataset == 'mycifar30':
        net = ResNet18(num_classes=args.class_per_task).to(device)
        normalizer = InputNormalize(
            new_mean=torch.tensor([0.5071, 0.4867, 0.4408]),
            new_std=torch.tensor([0.2675, 0.2565, 0.2761]),
            device=device
        )
    elif args.dataset in ['digit5', 'digit4']:
        net = AlexNet(num_classes=args.class_per_task).to(device)
        normalizer = InputNormalize(
            new_mean=torch.tensor([0.5, 0.5, 0.5]),
            new_std=torch.tensor([0.5, 0.5, 0.5]),
            device=device
        )
    else:
        raise NameError("illegal dataset.")
        
    max_epoch = args.epochs

    criterion = torch.nn.CrossEntropyLoss()
    memory = None
    for sess in range(args.num_task):
        
        w_logger.log(f"train sess{sess}...")
        
        task_info, mytrainloader, mytestloader, for_memory = inc_dataset.new_task(memory) 
        memory = inc_dataset.get_memory(memory, for_memory, sess)

        if sess > 0 and args.kd:
            model = deepcopy(net)
            if args.resume and sess == 1:
                first_task_weight_path = args.ckpt
            else:
                first_task_weight_path =  save_path / f'sess{(sess-1)}_last.pth'
            state_dict = torch.load(str(first_task_weight_path), map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

        if sess == 0 and args.resume:
            state_dict = torch.load(args.ckpt, map_location=device)
            net.load_state_dict(state_dict)
            Acc, task_acc = test(0)
            acc_dict['task'].append(float(f"{Acc:.4f}"))
            acc_dict[0].append(float(
                f"{task_acc[int(args.task_order[0])]:.4f}"
            ))
            continue
        
        cur_lr = args.lr
        for epoch in range(max_epoch):
            if args.decay == 'linear':
                cur_lr = get_linea_lr(epoch, args.schedule, cur_lr)
            elif args.decay == 'sgdr':
                cur_lr = get_sgdr_lr(epoch=epoch, eta_max=0.1)
            if args.optim == 'sgdm':

                # oracle, use all data to train model.
                if args.memory_size == 0:
                    if sess > 0:
                        if args.kd:
                            if args.dataset in ['digit5','digit4']:   # digit5: kd,baseline using 1e-3
                                cur_lr = 1e-3
                        else:
                            cur_lr = 1e-3     # baseline --> use a small lr

                optimizer = optim.SGD(net.parameters(), cur_lr, momentum=0.9, weight_decay=1e-4)
            elif args.optim == 'adam':
                optimizer = optim.Adam(net.parameters(), lr=cur_lr)
            
            train(epoch)
            Acc, task_acc = test(epoch)
            w_logger.log("\n")

            if epoch+1 == max_epoch:
                torch.save(net.state_dict(), str(save_path/f"sess{sess}_last.pth"))
                acc_dict['task'].append(float(f"{Acc:.4f}"))
                for i in range(sess+1):
                    acc_dict[i].append(float(
                        f"{task_acc[int(args.task_order[i])]:.4f}"
                    ))
                print(acc_dict)

    compute_acc_fgt(acc_dict, args.num_task)
    
    json_f = save_path / 'AccInfo.json'
    with open(json_f, 'w') as f:
        json.dump(acc_dict, f, indent=2)










