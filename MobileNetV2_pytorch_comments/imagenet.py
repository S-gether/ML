'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# 1) 구 파이썬 버전에서 미래 버전 기능 사용
from __future__ import print_function

# 2) 기본 라이브러리
import argparse
import os
import random
import shutil
import time
import warnings

# 3) torch 라이브러리
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from tensorboardX import SummaryWriter

# 4) 사용자 제작 라이브러리
import models.imagenet as customized_models
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.dataloaders import *

# ==============================================================
# 1. Model Name 불러오기
# ==============================================================
# 1) torchvision.models에서의 Model Name 종류
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 2) 사용자 정의 Model Name 종류
customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

# 3) 전체 model 종류 = torchvision.models에서의 Model Name 종류 + 사용자 정의 Model Name 종류
for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]
model_names = default_model_names + customized_models_names

# ==============================================================
# 1. 변수 선언
# ==============================================================
# 0) Argparse 선언
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# --------------------------------------------------------------
# 1-1) Distributed Training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
# --------------------------------------------------------------
# 1-2) Resume Checkpoint Saved Path
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# 1-3) Transfer Learning - Pre Trained Model 사용 유무
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# 1-4) Transfer Learning - Saved Path
parser.add_argument('--weight', default='mobilenetv2_1.0-0c6065bc.pth', type=str, metavar='WEIGHT', # 기존 : ''
                    help='path to pretrained weight (default: none)')

# 2-1) Dataset Path
parser.add_argument('-d', '--data', metavar='DIR', default='/home/hi/Datasets/ILSVRC2012/', # 기존 : default 존재하지 않음
                    help='path to dataset')
# --------------------------------------------------------------
# 2-2) BackEnd - DALI 사용 유무 결정
parser.add_argument('--data-backend', metavar='BACKEND', default='pytorch',
                    choices=DATA_BACKEND_CHOICES)
# --------------------------------------------------------------
# 2-3) DataLoader Worker 개수
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# 2-4) DataLoader Batch Size
parser.add_argument('-b', '--batch-size', default=64, type=int, # 기존 : 256
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# 3-1) Model 종류
parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenetv2', # 기존 : resnet18
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: mobilenetv2)')
# 3-2) Model - Input Size
parser.add_argument('--input-size', type=int, default=224, help='MobileNet model input resolution')
# 3-3) Model - Width Multiplier
parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')

# 4-1) Optimization - Momentum
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# 4-2) Optimization - Weight Decay
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float, # 기존 : 1e-4
                    metavar='W', help='weight decay (default: 4e-5)',
                    dest='weight_decay')
# 4-3) Optimzation - Learning Rate
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, # 기존 : 0.1
                    metavar='LR', help='initial learning rate', dest='lr')
# 4-4) Scheduler
parser.add_argument('--lr-decay', type=str, default='cos', # 기존 : step
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

# 5-1) Train Start Epochs
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# 5-2) Train Epochs
parser.add_argument('--epochs', default=150, type=int, metavar='N', # 기존 : 90
                    help='number of total epochs to run')
# 5-3) Train Seed
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# 6) Checkpoint - Save Path
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

# 7) Evaluate
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, # 기존 : action='store_true', default는 없음
                    help='evaluate model on validation set')

# 8) Visualization
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

# ==============================================================
# 2. Main문
# ==============================================================
best_prec1 = 0

def main():
    # --------------------------------------------------------------
    # 1) 변수 불러오기 + 잘라내기
    # --------------------------------------------------------------
    global args, best_prec1
    args = parser.parse_args()
    # print("args", args) => Namespace(world_size=-1, rank=-1, dist_url='tcp://224.66.41.62:23456', dist_backend='nccl', resume='', pretrained=False, weight='mobilenetv2_1.0-0c6065bc.pth', data='/home/hi/Datasets/ILSVRC2012/', data_backend='pytorch', workers=4, batch_size=64, arch='mobilenetv2', input_size=224, width_mult=1.0, momentum=0.9, weight_decay=4e-05, lr=0.05, lr_decay='cos', step=30, schedule=[150, 225], gamma=0.1, warmup=False, start_epoch=0, epochs=150, seed=None, checkpoint='checkpoints', evaluate=True, print_freq=10)

    # --------------------------------------------------------------
    # 2) Seed 정의
    # --------------------------------------------------------------
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # 3) 분산 학습 설정
    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    # --------------------------------------------------------------
    # 4) Model 정의
    # --------------------------------------------------------------
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](width_mult=args.width_mult)
    # print("models.__dict__", models.__dict__) => {'__name__': 'torchvision.models', '__doc__': None, '__package__': 'torchvision.models', '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7f465b6194f0>, '__spec__': ModuleSpec(name='torchvision.models', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7f465b6194f0>, origin='/home/hi/anaconda3/envs/GSC/lib/python3.9/site-packages/torchvision/models/__init__.py', submodule_search_locations=['/home/hi/anaconda3/envs/GSC/lib/python3.9/site-packages/torchvision/models']), '__path__': ['/home/hi/anaconda3/envs/GSC/lib/python3.9/site-packages/torchvision/models'], '__file__': '/home/hi/anaconda3/envs/GSC/lib/python3.9/site-packages/torchvision/models/__init__.py', '__cached__': '/home/hi/anaconda3/envs/GSC/lib/python3.9/site-packages/torchvision/models/__pycache__/__init__.cpython-39.pyc', '__builtins__': {'__name__': 'builtins', '__doc__': "Built-in functions, exceptions, and other objects.\n\nNoteworthy: None is the `nil' object; Ellipsis represents `...' in slices.", '__package__': '', '__loader__': <class '_frozen_importlib.BuiltinImporter'>, '__spec__': ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>, origin='built-in'), '__build_class__': <built-in function __build_class__>, '__import__': <built-in function __import__>, 'abs': <built-in function abs>, 'all': <built-in function all>, 'any': <built-in function any>, 'ascii': <built-in function ascii>, 'bin': <built-in function bin>, 'breakpoint': <built-in function breakpoint>, 'callable': <built-in function callable>, 'chr': <built-in function chr>, 'compile': <built-in function compile>, 'delattr': <built-in function delattr>, 'dir': <built-in function dir>, 'divmod': <built-in function divmod>, 'eval': <built-in function eval>, 'exec': <built-in function exec>, 'format': <built-in function format>, 'getattr': <built-in function getattr>, 'globals': <built-in function globals>, 'hasattr': <built-in function hasattr>, 'hash': <built-in function hash>, 'hex': <built-in function hex>, 'id': <built-in function id>, 'input': <built-in function input>, 'isinstance': <built-in function isinstance>, 'issubclass': <built-in function issubclass>, 'iter': <built-in function iter>, 'len': <built-in function len>, 'locals': <built-in function locals>, 'max': <built-in function max>, 'min': <built-in function min>, 'next': <built-in function next>, 'oct': <built-in function oct>, 'ord': <built-in function ord>, 'pow': <built-in function pow>, 'print': <built-in function print>, 'repr': <built-in function repr>, 'round': <built-in function round>, 'setattr': <built-in function setattr>, 'sorted': <built-in function sorted>, 'sum': <built-in function sum>, 'vars': <built-in function vars>, 'None': None, 'Ellipsis': Ellipsis, 'NotImplemented': NotImplemented, 'False': False, 'True': True, 'bool': <class 'bool'>, 'memoryview': <class 'memoryview'>, 'bytearray': <class 'bytearray'>, 'bytes': <class 'bytes'>, 'classmethod': <class 'classmethod'>, 'complex': <class 'complex'>, 'dict': <class 'dict'>, 'enumerate': <class 'enumerate'>, 'filter': <class 'filter'>, 'float': <class 'float'>, 'frozenset': <class 'frozenset'>, 'property': <class 'property'>, 'int': <class 'int'>, 'list': <class 'list'>, 'map': <class 'map'>, 'object': <class 'object'>, 'range': <class 'range'>, 'reversed': <class 'reversed'>, 'set': <class 'set'>, 'slice': <class 'slice'>, 'staticmethod': <class 'staticmethod'>, 'str': <class 'str'>, 'super': <class 'super'>, 'tuple': <class 'tuple'>, 'type': <class 'type'>, 'zip': <class 'zip'>, '__debug__': True, 'BaseException': <class 'BaseException'>, 'Exception': <class 'Exception'>, 'TypeError': <class 'TypeError'>, 'StopAsyncIteration': <class 'StopAsyncIteration'>, 'StopIteration': <class 'StopIteration'>, 'GeneratorExit': <class 'GeneratorExit'>, 'SystemExit': <class 'SystemExit'>, 'KeyboardInterrupt': <class 'KeyboardInterrupt'>, 'ImportError': <class 'ImportError'>, 'ModuleNotFoundError': <class 'ModuleNotFoundError'>, 'OSError': <class 'OSError'>, 'EnvironmentError': <class 'OSError'>, 'IOError': <class 'OSError'>, 'EOFError': <class 'EOFError'>, 'RuntimeError': <class 'RuntimeError'>, 'RecursionError': <class 'RecursionError'>, 'NotImplementedError': <class 'NotImplementedError'>, 'NameError': <class 'NameError'>, 'UnboundLocalError': <class 'UnboundLocalError'>, 'AttributeError': <class 'AttributeError'>, 'SyntaxError': <class 'SyntaxError'>, 'IndentationError': <class 'IndentationError'>, 'TabError': <class 'TabError'>, 'LookupError': <class 'LookupError'>, 'IndexError': <class 'IndexError'>, 'KeyError': <class 'KeyError'>, 'ValueError': <class 'ValueError'>, 'UnicodeError': <class 'UnicodeError'>, 'UnicodeEncodeError': <class 'UnicodeEncodeError'>, 'UnicodeDecodeError': <class 'UnicodeDecodeError'>, 'UnicodeTranslateError': <class 'UnicodeTranslateError'>, 'AssertionError': <class 'AssertionError'>, 'ArithmeticError': <class 'ArithmeticError'>, 'FloatingPointError': <class 'FloatingPointError'>, 'OverflowError': <class 'OverflowError'>, 'ZeroDivisionError': <class 'ZeroDivisionError'>, 'SystemError': <class 'SystemError'>, 'ReferenceError': <class 'ReferenceError'>, 'MemoryError': <class 'MemoryError'>, 'BufferError': <class 'BufferError'>, 'Warning': <class 'Warning'>, 'UserWarning': <class 'UserWarning'>, 'DeprecationWarning': <class 'DeprecationWarning'>, 'PendingDeprecationWarning': <class 'PendingDeprecationWarning'>, 'SyntaxWarning': <class 'SyntaxWarning'>, 'RuntimeWarning': <class 'RuntimeWarning'>, 'FutureWarning': <class 'FutureWarning'>, 'ImportWarning': <class 'ImportWarning'>, 'UnicodeWarning': <class 'UnicodeWarning'>, 'BytesWarning': <class 'BytesWarning'>, 'ResourceWarning': <class 'ResourceWarning'>, 'ConnectionError': <class 'ConnectionError'>, 'BlockingIOError': <class 'BlockingIOError'>, 'BrokenPipeError': <class 'BrokenPipeError'>, 'ChildProcessError': <class 'ChildProcessError'>, 'ConnectionAbortedError': <class 'ConnectionAbortedError'>, 'ConnectionRefusedError': <class 'ConnectionRefusedError'>, 'ConnectionResetError': <class 'ConnectionResetError'>, 'FileExistsError': <class 'FileExistsError'>, 'FileNotFoundError': <class 'FileNotFoundError'>, 'IsADirectoryError': <class 'IsADirectoryError'>, 'NotADirectoryError': <class 'NotADirectoryError'>, 'InterruptedError': <class 'InterruptedError'>, 'PermissionError': <class 'PermissionError'>, 'ProcessLookupError': <class 'ProcessLookupError'>, 'TimeoutError': <class 'TimeoutError'>, 'open': <built-in function open>, 'quit': Use quit() or Ctrl-D (i.e. EOF) to exit, 'exit': Use exit() or Ctrl-D (i.e. EOF) to exit, 'copyright': Copyright (c) 2001-2022 Python Software Foundation.

    # --------------------------------------------------------------
    # 5) Model to Cuda
    # --------------------------------------------------------------
    # (1) 분산 학습 아닌 경우
    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # (2) 분산 학습인 경우
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    # --------------------------------------------------------------
    # 6) Loss Function + Optimizer 정의
    # --------------------------------------------------------------
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # --------------------------------------------------------------
    # 7) Save Checkpoint Dir
    # --------------------------------------------------------------
    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # --------------------------------------------------------------
    # 8) Logger 설정 + Resume Checkpoint
    # --------------------------------------------------------------
    # (1) Resume 설정한 경우 : Resume Checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # 1] Resume Checkpoint (Start Epoch / Best Accuracy / Model / StateDict)
            checkpoint = torch.load(args.resume) # 구조!!!
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # 2] Logger 열기
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True) # 구조!!!
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # (2) Resume 설정하지 않은 경우 : Start Checkpoint
    else:
        # 1] Logger 열기
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # --------------------------------------------------------------
    # 9) Cudnn 성능 최적화
    # --------------------------------------------------------------
    cudnn.benchmark = True

    # --------------------------------------------------------------
    # 10) DataLoader
    # --------------------------------------------------------------
    # (1) Data 불러오는 형식
    # 1] Data Backend == Pytorch인 경우
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    # 2] Data Backend == dali-gpu인 경우
    elif args.data_backend == 'dali-gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    # 3] Data Backend == dali-cpu인 경우
    elif args.data_backend == 'dali-cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()

    # (2) Data 불러오기
    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)
    val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)

    # --------------------------------------------------------------
    # 11) Evaluate : Pretrained Weight 불러오기 -> 평가
    # --------------------------------------------------------------
    if args.evaluate:
        from collections import OrderedDict
        # (1) StateDict 불러오기
        # 1] Weight 파일 존재하는 경우 : 불러온다
        if os.path.isfile(args.weight):
            print("=> loading pretrained weight '{}'".format(args.weight))
            source_state = torch.load(args.weight) # 구조!!!
            target_state = OrderedDict()
            for k, v in source_state.items(): # 구조!!!
                if k[:7] != 'module.':
                    k = 'module.' + k
                target_state[k] = v
            model.load_state_dict(target_state)
        # 2] Weight 파일 존재하지 않는 경우 : 불러오지 않는다
        else:
            print("=> no weight found at '{}'".format(args.weight))

        # (2) Evaluate
        validate(val_loader, val_loader_len, model, criterion)
        return

    # --------------------------------------------------------------
    # 12) Train & Update Logger & Update Checkpoint
    # --------------------------------------------------------------
    # (1) Checkpoint Path의 logs 정보 불러오기
    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs')) # 구조!!!

    # (2) Epoch 순회 (Update Train / Update Evaluate / LR / Save Checkpoint)
    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

        # 1] Train -> loss, accuracy 출력
        train_loss, train_acc = train(train_loader, train_loader_len, model, criterion, optimizer, epoch)

        # 2] Evaluate -> loss, accuracy 출력
        val_loss, prec1 = validate(val_loader, val_loader_len, model, criterion)

        lr = optimizer.param_groups[0]['lr'] # 구조!!!

        # 3] logger 저장 (learning rate / train_loss / val_loss / train_accuracy / val_accuracy)
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # 4] Tensorboard 저장 (learning rate / train_loss / val_loss / train_accuracy / val_accuracy)
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)

        # 5] Save Checkpoint (Epoch / Model Name / StateDict / Best Accuracy / Optimizer)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint) # 구조!!!

    # (3) Visualization(Close Logger / Close Writer / Save Checkpoint)
    logger.close()
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps')) # 구조!!!
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    writer.close()
    print('Best accuracy:')
    print(best_prec1)

# --------------------------------------------------------------
# 12) Train & Update Logger & Update Checkpoint
    # (2) Epoch 순회 (Update Train / Update Evaluate / LR / Save Checkpoint)
        # 1] Train -> loss, accuracy 출력
# --------------------------------------------------------------
def train(train_loader, train_loader_len, model, criterion, optimizer, epoch):
    # [1] Meter 변수 생성
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # [2] Train model
    model.train()

    # [3] Train 계산 + 업데이트 (Learning Rate / 소요 시간 / Output / Loss / Accuracy / Gradient Update)
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # 1]] Learning Rate 수정
        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # 2]] Loading 시간 측정
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # 3]] Output 확률값 추출
        output = model(input) # output : shape(64, 1000)
        # 4]] Loss 계산
        loss = criterion(output, target)
        # 5]] Accuracy + Loss 계산
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # 6]] Gradient 계산 + SGD Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7]] Batch Data 당 학습 소요시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

    # [4] 평균 Loss / Accuracy 반환
    return (losses.avg, top1.avg)

# --------------------------------------------------------------
# 12) Train & Update Logger & Update Checkpoint
    # (2) Epoch 순회 (Update Train / Update Evaluate / LR / Save Checkpoint)
        # 2] Evaluate -> loss, accuracy 출력
# --------------------------------------------------------------
def validate(val_loader, val_loader_len, model, criterion):
    # [1] Meter 변수 생성
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # [2] Evaluate model
    model.eval()

    # [3] Evaluate 계산 + 업데이트 (Learning Rate / 소요 시간 / Output / Loss / Accuracy / Gradient Update)
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # 1]] Data Loading 소요시간 측정
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # 2]] Output 확률값 추출 + Loss 계산 (Gradient Update 하지 않음)
        with torch.no_grad():
            output = model(input) # output : shape(64, 1000)
            # print("output[1].sum() : ", output[1].sum()) # temp : tensor(-3.5930e-08, device='cuda:0')
            loss = criterion(output, target) # loss : 6.9078 / shape(1)

        # 3]] Accuracy + Loss 종합적 계산
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # 4]] Batch Data 당 학습 소요시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

    # [4] 평균 Loss / Accuracy 반환
    return (losses.avg, top1.avg)

# --------------------------------------------------------------
# 12) Train & Update Logger & Update Checkpoint
    # (2) Epoch 순회 (Update Train / Update Evaluate / LR / Save Checkpoint)
        # 5] Save Checkpoint (Epoch / Model Name / StateDict / Best Accuracy / Optimizer)
# --------------------------------------------------------------
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    # [1] Save Checkpoint (파일 경로 : ./checkpoint/checkpoint.pth.tar)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # [2] Copy Best Checkpoint (파일 경로 : ./checkpoint/model_best.pth.tar)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

# --------------------------------------------------------------
# 12) Train & Update Logger & Update Checkpoint
    # (2) Epoch 순회 (Update Train / Update Evaluate / LR / Save Checkpoint)
        # 1] Train -> loss, accuracy 출력
            # [3] Train 계산 + 업데이트 (Learning Rate / 소요 시간 / Output / Loss / Accuracy / Gradient Update)
                # 1]] Learning Rate 수정
# --------------------------------------------------------------
from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    # 1. Learning Rate 불러오기
    lr = optimizer.param_groups[0]['lr']
    print("lr : ", lr)

    # 2. Learning Rate 조정 기준 설정 (Warm Iter 기준 / Current Iter 기준 / Max Iter 기준)
    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter # !!!
    max_iter = args.epochs * num_iter

    # 3. Learning Rate Decay 종류에 따른 Learning Rate 조정
    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter))) # gamma : 0.1
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    # 4. Learning Rate 저장
    for param_group in optimizer.param_groups: # 이거 왜 이렇게 했지???
        param_group['lr'] = lr

# 1. Main문
if __name__ == '__main__':
    main()
