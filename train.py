import torch
import torch.nn as nn
from torch.distributions import normal
import torchvision.models as models
from torch import optim
from torch.optim import lr_scheduler
import utils
import Network
from tqdm import tqdm
import os
import argparse
import datetime
from torchvision import datasets
import time
import random
import numpy as np
from SelfSupervised import ProxyTaskInfo
import string
import math

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['pretrain','supervised', 'test'], default='train')
parser.add_argument('--ss_task', choices=['rotation','exemplar', 'jigsaw', 'none'], default='none')
parser.add_argument('--init_cond', choices=['rotation', 'imagenet', 'random'], default='random')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--dataset', type=str, choices='pcam, BACH', default='pcam')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--model', choices=['resnet34'], default='resnet34')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_steps', default=[5], nargs='+', type=int)
parser.add_argument('--wd', type=float, default=1e-3)
parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--tqdm_off', action='store_true', default=False)
parser.add_argument('--dataset_path', type=str, default='/mnt/datasets/')




args = parser.parse_args()

if args.mode == "pretrain":
    if args.ss_task == "none":
        raise ValueError('To complete self supervised pretraining a pretraining task must be included')
else:
    if args.ss_task != "none":
        raise ValueError('No self supervised pretraining task should be included for supervised learning task')


#Set all random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)



if args.tqdm_off:
    def nop(it, *a, **k):
        return it
    tqdm = nop

if args.dataset == "pcam":
    dataset = utils.__dict__['ImageDataset_pcam']
    args.dataset_path = args.dataset_path + '/pcam/'
if args.dataset =='BACH':
    dataset = utils.__dict__['ImageDataset_BACH']
    args.dataset_path = args.dataset_path + '/bach/'
    args.batch_size=32


def save_checkpoint():
    checkpoint = [model.state_dict(), opt.state_dict()]
    torch.save(checkpoint, '%s/checkpoint_%d_%d.pth' % (save_path, args.seed, epoch))
    torch.save(checkpoint, '%s/checkpoint_%d_%d.pth' % (save_mostrecent, args.seed, epoch))


def save_best_checkpoint():
    checkpoint = [model.state_dict(), opt.state_dict()]
    torch.save(checkpoint, '%s/checkpoint_best_%d.pth' % (save_path, args.seed))
    torch.save(checkpoint, '%s/checkpoint_best_%d.pth' % (save_mostrecent, args.seed))


def load_checkpoint(load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint[0])
    opt.load_state_dict(checkpoint[1])

def compute_acc(class_out, targets):
    preds = torch.max(class_out, 1)[1]
    softmax  = torch.exp(class_out[0])
    pos = 0; 
    for ix in range(preds.size(0)):
        if preds[ix] == targets[ix]:
            pos = pos + 1
    accuracy = pos / preds.size(0) * 100

    return accuracy

def train():
    model.train()
    avg_loss = 0
    avg_acc = 0
    avg_real_acc = 0
    avg_fake_acc = 0
    count = 0
    for _, (data, target) in enumerate(tqdm(train_data_loader)):
        opt.zero_grad()
        data, target  = data.cuda(), target.long().cuda()
        out = model(data)
        loss = ent_loss(out, target)
        loss.backward()
        opt.step()
        avg_loss = avg_loss + loss.item()
        curr_acc = compute_acc(out.data, target.data)
        avg_acc = avg_acc + curr_acc
        count = count + 1
    avg_loss = avg_loss / count
    avg_acc = avg_acc / count
    print('Epoch: %d; Loss: %f; Acc: %.2f; ' % (epoch, avg_loss, avg_acc))
    loss_logger.log(str(avg_loss))
    acc_logger.log(str(avg_acc))
    return avg_loss

def test():
    print('Testing')
    model.eval()
  
    pos=0; total=0;
    prediction_list = []
    groundtruth_list = []
    for _, (data, target) in enumerate(tqdm(test_data_loader)):
        data, target  = data.cuda(), target.long().cuda()
        with torch.no_grad():
            out = model(data)
        pred = torch.max(out, out.dim() - 1)[1]
        pos = pos + torch.eq(pred.cpu().long(), target.data.cpu().long()).sum().item()
        
        groundtruth_list += target.data.tolist()
        prediction_list += out[:,1].tolist()


        total = total + data.size(0)
    acc = pos * 1.0 / total * 100
    print('Acc: %.2f' % acc)

    return acc

def create_model(num_classes, pretrain_path):
    # Create a model with a specific initialization
    if args.mode =='test':
        Model = Network.__dict__['Model_Random']
        model = Model(num_classes=num_classes)

    else:
        if args.init_cond == 'imagenet':
            Model = Network.__dict__['Model_ImageNet']
            model = Model(num_classes=num_classes)

        if args.init_cond == 'random':
            Model = Network.__dict__['Model_Random']
            model = Model(num_classes=num_classes)


        if args.init_cond == 'rotation':
            Model = Network.__dict__['Model_Rotation']
            model = Model(num_classes=num_classes, pretrain_path=load_ss)

    model = model.cuda()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = lr_scheduler.MultiStepLR(opt, milestones=args.lr_steps, gamma=0.1)
    return model, opt, sch

if __name__ == "__main__":

    #set up directory to save files
    save_pathroot = 'results/%s/%s' % (args.dataset, args.mode)
    if args.mode =='pretrain':
        task=args.ss_task
    else:
        task=args.init_cond

    save_path = save_pathroot + '/' + args.model + '/' + task + '/' + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H-%M")
    save_mostrecent = save_pathroot + '/' + args.model + '/' + task + '/mostrecent'

    if args.mode =='supervised':
        load_ss = save_mostrecent.replace('supervised', 'pretrain') + '/checkpoint_best_1111.pth'
    if args.mode == 'test':
        load_test = save_mostrecent.replace('test', 'supervised') + '/checkpoint_best_1111.pth'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        if not os.path.exists(save_mostrecent):
            os.makedirs(save_mostrecent)

    #Load in data for training/validation or testing
    if not args.mode == 'test':
        train_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, train=True, is_test=False, ss_task=args.ss_task), batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, train=False, is_test=False , ss_task=args.ss_task), batch_size=args.batch_size, num_workers=4)

    else:
        test_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, train=False, is_test=True , ss_task=args.ss_task), batch_size=args.batch_size, num_workers=4)

    #Pull specs for model depending on in if it is pretraining, supervised, or test

    if task =="rotation" and args.mode!='test':
        num_classes, pretrain_path = ProxyTaskInfo.rotation(args.dataset)
    else:
        num_classes, pretrain_path = ProxyTaskInfo.supervised(args.dataset)


    #Create the model
    print(num_classes)
    model, opt, sch = create_model(num_classes=num_classes, pretrain_path=pretrain_path)

    if not args.mode=='test':
        loss_logger = utils.TextLogger('loss', '{}/loss_{}.log'.format(save_path, args.seed))
        acc_logger = utils.TextLogger('acc', '{}/acc_{}.log'.format(save_path, args.seed))
        test_acc_logger = utils.TextLogger('test_acc', '{}/test_acc_{}.log'.format(save_path, args.seed))


    ent_loss = nn.CrossEntropyLoss().cuda()
    epoch = 1

    if args.load_epoch != -1:
        epoch = args.load_epoch + 1
        load_checkpoint('%s/checkpoint_%d_%d.pth' % (save_path, args.seed, args.load_epoch))

    if not args.mode=='test':
        best_acc = 0
        while True:
            loss = train()
            print(opt.param_groups[0]['lr'])
            sch.step(epoch)
            acc = test()

            test_acc_logger.log(str(acc))


            if epoch % args.save_epoch == 0:
                save_checkpoint()
            if acc > best_acc:
                best_acc = acc
                save_best_checkpoint()

            if epoch == args.max_epochs:
                break

            epoch += 1
    else:
        load_checkpoint(load_test)
        test()
