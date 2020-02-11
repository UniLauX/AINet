import time
import argparse
import torch.optim as optim
import sys
sys.path.append('./models/')
from models_lw_3D import *
from functions_for_training import *
from torch.nn import DataParallel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Training settings
parser = argparse.ArgumentParser(description='HSI classification')
parser.add_argument('--dataset', type=str, default='PaviaU')
parser.add_argument('--model_name', type=str, default='LWNet_3')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--restore', type=bool, default=True)
parser.add_argument('--transfer', type=bool, default=True)
parser.add_argument('--transfer_model', type=str, default='Pavia_model')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--test_batch_size', type=int, default=80)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seed', default=1)
parser.add_argument('--model_save_interval', type=int, default=1)
args = parser.parse_args()

train_data_txt = '../data_preprocess/data_list/{}_train.txt'.format(args.dataset)
test_data_txt = '../data_preprocess/data_list/{}_test.txt'.format(args.dataset)
transfer_model_dir = './transfer_model/{}.pkl'.format(args.transfer_model)

trained_model_dir = './LWNet_t_{}_2_{}/'.format(args.transfer_model, args.dataset) + args.model_name + '/'
train_info_record = trained_model_dir + 'train_info_' + args.model_name + '.txt'

torch.manual_seed(args.seed)

if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# data loaders
train_loader = torch.utils.data.DataLoader(
    data_loader(train_data_txt)
    , batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    data_loader(test_data_txt)
    , batch_size=args.test_batch_size)

make_if_not_exist(trained_model_dir)

if args.dataset == 'PaviaU':
    num_cla=9
elif args.dataset == 'Indian':
    num_cla=16
else:
    num_cla=13

model = DataParallel(dict[args.model_name](num_classes=num_cla, dropout_keep_prob=0))
if args.use_cuda:
    model.cuda()

if args.transfer:
    optimizer = optim.SGD([{'params': model.module.conv.parameters(), 'lr': args.lr*0.1}, {'params': model.module.fc.parameters(), 'lr': args.lr}], momentum=args.momentum, weight_decay=1e-5)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)

start_epoch = 0
if args.restore and len(os.listdir(trained_model_dir)):
    model, start_epoch = model_restore(model, trained_model_dir)
elif not os.listdir(trained_model_dir) and args.transfer:
    state_dict_transfer_model = (torch.load(transfer_model_dir))
    # replace the parameters of the fully connected parts in transfer model with that of initialization model
    state_dict_transfer_model['module.fc.weight']=model.state_dict()['module.fc.weight']
    state_dict_transfer_model['module.fc.bias'] = model.state_dict()['module.fc.bias']
    model.load_state_dict(state_dict_transfer_model)

train_info_record = trained_model_dir + 'train_info_' + args.model_name + '.txt'

for epoch in range(start_epoch+1, args.epochs+1):
    start = time.time()
    train(epoch, model, train_loader, optimizer, args)
    end = time.time()
    print('epoch: {} , cost {} seconds'.format(epoch,  end-start))

    if epoch % args.model_save_interval == 0 and epoch >=50:
        model_name = trained_model_dir + '/trained_model{}.pkl'.format(epoch)
        torch.save(model.cpu().state_dict(), model_name)
        if args.use_cuda: model.cuda()
        train_loss, train_acc = val(model, train_loader, args)
        print('train_loss: {:.4f}, train_acc: {:.2f}%'.format(train_loss, train_acc))
        val_loss, val_acc = val(model, test_loader, args)
        print('val_loss: {:.4f}, val_acc: {:.2f}%'.format(val_loss, val_acc))
        with open(train_info_record, 'a') as f:
            f.write('timecost:{:.2f}, lr:{}, epoch:{}, train_loss:{:.4f}, train_acc:{:.2f}, val_loss:{:.6f}, val_acc:{:.2f}'.format(
                (end-start)/60, optimizer.param_groups[0]['lr'], epoch, train_loss, train_acc, val_loss, val_acc) + '\n'
            )

info_plot(train_info_record)
