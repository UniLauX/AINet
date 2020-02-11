import os
import time
import argparse
import torch.optim as optim
import sys
sys.path.append('../models/')
from models.ARNets import *
from functions_for_training import train, val, make_if_not_exist, model_restore
from data_loader_for_2HSIs_pretrain import data_loader
from torch.nn import DataParallel


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Training settings
parser = argparse.ArgumentParser(description='HSI classification')
parser.add_argument('--dataset_1', type=str, default='KSC')
parser.add_argument('--category_start', type=int, default=13)
parser.add_argument('--dataset_2', type=str, default='PaviaU')
parser.add_argument('--model_name', type=str, default='ARNet_2')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--restore', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=18)
parser.add_argument('--test_batch_size', type=int, default=40)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seed', default=1)
parser.add_argument('--model_save_interval', type=int, default=2)
args = parser.parse_args()

train_data_dir_1 = '../data_preprocess/data_list/{}_train.txt'.format(args.dataset_1)
test_data_dir_1 = '../data_preprocess/data_list/{}_test.txt'.format(args.dataset_1)
train_data_dir_2 = '../data_preprocess/data_list/{}_train.txt'.format(args.dataset_2)
test_data_dir_2 = '../data_preprocess/data_list/{}_test.txt'.format(args.dataset_2)

trained_model_dir = './train_{}_{}/'.format(args.dataset_1, args.dataset_2) + args.model_name + '/'
train_info_record_1 = trained_model_dir + '{}_train_info_'.format(args.dataset_1) + args.model_name + '.txt'
train_info_record_2 = trained_model_dir + '{}_train_info_'.format(args.dataset_2) + args.model_name + '.txt'



torch.manual_seed(args.seed)

if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# data loaders
train_loader_1 = torch.utils.data.DataLoader(
    data_loader(train_data_dir_1, 0)
    , batch_size=args.batch_size, shuffle=True)
train_loader_2 = torch.utils.data.DataLoader(
    data_loader(train_data_dir_2, args.category_start)
    , batch_size=args.batch_size, shuffle=True)
test_loader_1 = torch.utils.data.DataLoader(
    data_loader(test_data_dir_1)
    , batch_size=args.test_batch_size)
test_loader_2 = torch.utils.data.DataLoader(
    data_loader(test_data_dir_2, args.category_start)
    , batch_size=args.test_batch_size)

make_if_not_exist(trained_model_dir)
model = DataParallel(dict[args.model_name](num_classes=22, dropout_keep_prob=0))
if args.use_cuda:
    model=model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)

start_epoch = 0
if args.restore and len(os.listdir(trained_model_dir)):
    model, start_epoch = model_restore(model, trained_model_dir)

for epoch in range(start_epoch+1, args.epochs+1):
    start = time.time()
    train(epoch, model, train_loader_1, optimizer, args)
    train(epoch, model, train_loader_2, optimizer, args)
    end = time.time()
    print('epoch: {} , cost {} seconds'.format(epoch,  end-start))

    if epoch % args.model_save_interval == 0 and epoch > args.epochs*0.6:
        model_name = trained_model_dir + '/trained_model{}.pkl'.format(epoch)
        torch.save(model.cpu().state_dict(), model_name)
        if args.use_cuda: model.cuda()
        train_loss_1, train_acc_1 = val(model, train_loader_1, args)
        print('train_loss_1: {:.4f}, train_acc_1: {:.2f}%'.format(train_loss_1, train_acc_1))
        train_loss_2, train_acc_2= val(model, train_loader_2, args)
        print('train_loss_2: {:.4f}, train_acc_2: {:.2f}%'.format(train_loss_2, train_acc_2))
        val_loss_1, val_acc_1 = val(model, test_loader_1, args)
        print('val_loss_1: {:.4f}, val_acc_1: {:.2f}%'.format(val_loss_1, val_acc_1))
        val_loss_2, val_acc_2 = val(model, test_loader_2, args)
        print('val_loss_2: {:.4f}, val_acc_2: {:.2f}%'.format(val_loss_2, val_acc_2))
        with open(train_info_record_1, 'a') as f:
            f.write('timecost:{:.2f}, lr:{}, epoch:{}, train_loss:{:.4f}, train_acc:{:.2f}, val_loss:{:.6f}, val_acc:{:.2f}'.format(
                (end-start)/60, optimizer.param_groups[0]['lr'], epoch, train_loss_1, train_acc_1, val_loss_1, val_acc_1) + '\n'
            )
        with open(train_info_record_2, 'a') as f:
            f.write('timecost:{:.2f}, lr:{}, epoch:{}, train_loss:{:.4f}, train_acc:{:.2f}, val_loss:{:.6f}, val_acc:{:.2f}'.format(
                (end-start)/60, optimizer.param_groups[0]['lr'], epoch, train_loss_2, train_acc_2, val_loss_2, val_acc_2) + '\n'
            )

info_plot(train_info_record_1)
info_plot(train_info_record_2)

