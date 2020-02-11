import time
import datetime
import argparse
import torch.optim as optim
import sys
sys.path.append('./models/')
from ARNets import *
from functions_for_training import *
from functions_for_evaluating import acc_calculation
from torch.nn import DataParallel

# Training settings
parser = argparse.ArgumentParser(description='HSI classification')
parser.add_argument('--dataset', type=str, default='PaviaU')
parser.add_argument('--model_name', type=str, default='ARNet_2')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--devices', type=str, default='0')
parser.add_argument('--restore', type=bool, default=False)
parser.add_argument('--transfer', type=bool, default=True)
parser.add_argument('--transfer_model', type=str, default='Pavia')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--test_batch_size', type=int, default=40)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seed', default=1)
parser.add_argument('--model_save_interval', type=int, default=1)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

train_data_txt = '../data_preprocess/data_list/{}_train.txt'.format(args.dataset)
test_data_txt = '../data_preprocess/data_list/{}_test.txt'.format(args.dataset)
transfer_model_dir = './transfer_model/{}.pkl'.format(args.transfer_model)

trained_model_dir = './ARNet_t_{}_2_{}/'.format(args.transfer_model, args.dataset) + args.model_name + '/'
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

if args.dataset in ['PaviaU', 'Pavia']:
    num_cla = 9
elif args.dataset in ['Indian', 'Salinas']:
    num_cla = 16
elif args.dataset == 'KSC':
    num_cla = 13
else:
    print('undefined dataset')

model = DataParallel(dict[args.model_name](num_classes=num_cla, dropout_keep_prob=0))
if args.use_cuda:
    model.cuda()

if args.transfer:
    optimizer = optim.SGD([{'params': model.module.conv.parameters(), 'lr': args.lr*0.1}, {'params': model.module.classifier.parameters(), 'lr': args.lr}], momentum=args.momentum, weight_decay=1e-5)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)

start_epoch = 0
if args.restore and len(os.listdir(trained_model_dir)):
    model, start_epoch = model_restore(model, trained_model_dir)
elif not os.listdir(trained_model_dir) and args.transfer:
    state_dict_transfer_model = (torch.load(transfer_model_dir))
    # replace the parameters of the fully connected parts in transfer model with that of initialization model
    state_dict_transfer_model['module.classifier.weight']=model.state_dict()['module.classifier.weight']
    state_dict_transfer_model['module.classifier.bias'] = model.state_dict()['module.classifier.bias']
    model.load_state_dict(state_dict_transfer_model)

train_info_record = trained_model_dir + 'train_info_' + args.model_name + '.txt'

for epoch in range(start_epoch+1, args.epochs+1):
    start = time.time()
    train(epoch, model, train_loader, optimizer, args)
    end = time.time()
    print('epoch: {} , cost {} seconds'.format(epoch,  end-start))    

samples_loader = torch.utils.data.DataLoader(
    data_loader(test_data_txt)
    , batch_size=args.test_batch_size, shuffle=False)
print('start to calculate acc')
acc = acc_calculation(model, samples_loader, args)
eval_info_dir = './ARNet_t_{}_2_{}.txt'.format(args.transfer_model, args.dataset)
now_time = datetime.datetime.now().strftime('%Y-%m-%d')
with open(eval_info_dir, 'a') as f:
    f.write('date:{} acc:{}\n'.format(now_time, acc))
# info_plot(train_info_record)
