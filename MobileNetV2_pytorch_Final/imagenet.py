'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
# ==============================================================
# 0. 라벨 정보
# ==============================================================
# 0 : 기쁨
# 1 : 당황
# 2 : 분노
# 3 : 불안
# 4 : 상처
# 5 : 슬픔
# 6 : 중립

# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# 1) 구 파이썬 버전에서 미래 버전 기능 사용
from __future__ import print_function

# 2) 기본 라이브러리
import argparse
import random
import warnings

# 3) torch 라이브러리
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed

# 4) 사용자 제작 라이브러리
import models.imagenet as customized_models
from utils.dataloaders import *

# ==============================================================
# 1. 변수 선언
# ==============================================================
# 0) Argparse 선언 ★★ 작동될까???
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# 1-1) Transfer Learning - Saved Model Path ★★ path 변경 / 작동될까???
parser.add_argument('--pretrained_model_path', default='/home/hi/Jupyter/MobileNetV2_pytorch_Final/checkpoints/model_best.pth_tar', type=str, metavar='WEIGHT',   # 기존 이름 : '--pretrained_model_path' / # transfer learng 사용 시 : default : 'mobilenetv2_1.0-0c6065bc.pth'
                    help='path to pretrained weight (default: none)')

# 2-1) Dataset Path ★★ path 변경 / 작동될까???
parser.add_argument('-d', '--data-path', metavar='DIR', default='/media/hi/SK Gold P31/Temp1/compression/', # 기존 : default 존재하지 않음 -> "/home/hi/Datasets/Cifar10/" -> "/media/hi/SK Gold P31/Temp/" -> 현재 : "/media/hi/SK Gold P31/Korean_Emotion_Movie/"
                    help='path to dataset')

# 2-2) DataLoader Worker 개수 ★★ worker 변경 / 작동될까???
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', # 기존 : 4(에러) -> 현재 : 8(최적)
                    help='number of data loading workers (default: 4)')
# 2-3) DataLoader Batch Size ★★ batch_size / 작동될까???
parser.add_argument('-b', '--batch-size', default=1, type=int, # 기존 : 256(에러) -> 128(에러) -> 현재 : 64(최적) -> 테스트 : 1
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# 3-1) Model - Input Size
parser.add_argument('--input-size', type=int, default=224, help='MobileNet model input resolution')
# 3-2) Model - Width Multiplier
parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')

# 5-3) Train Seed
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# ==============================================================
# 2. 함수 정의
# ==============================================================
# --------------------------------------------------------------
# 1) PIL <-> Numpy <-> Tensor 함수 정의
# --------------------------------------------------------------
def tensor_to_pltimg(tensor_image):
    return tensor_image.permute(1,2,0).numpy()

# --------------------------------------------------------------
# 2) Train & Update Logger & Update Checkpoint
    # (2) Epoch 순회 (Update Train / Update Evaluate / LR / Save Checkpoint)
        # 1] Train -> loss, accuracy 출력
            # [1] Meter 변수 생성
# --------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    # 1]] Value의 다양한 metric 계산
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# --------------------------------------------------------------
# 8) Evaluate : Pretrained Weight 불러오기 -> 평가
    # (2) Evaluate
        # 3] Evaluate 계산 + 업데이트 (Learning Rate / 소요 시간 / Output / Loss / Accuracy / Gradient Update)
            # [3] Accuracy + Loss 종합적 계산
# --------------------------------------------------------------
def accuracy(output, target):
    with torch.no_grad():
        # 1]] 초기 변수 설정 (Accuracy 계산할 개수 / BatchSize)
        maxk = 1
        batch_size = target.size(0) # batch_size : 1

        # 2]] model의 classifier 결과 중에서, Top 5개 예측값 내림차순 추출
        _, pred = output.topk(maxk, 1, True, True) # pred : shape(batch_size, maxk)
        target = target.view(1, -1) # target : shape(maxk, batch_size)
        pred = pred.t() # pred : shape(maxk, batch_size)

        # 3]] Top 1개 내림차순 정답 True/False 분류 (예측값 <-> 실제값 비교)
        correct = pred.eq(target.expand_as(pred)) # correct : shape(maxk, batch_size)
        # print(f"target : {target}, pred : {pred}")

        # 4]] Top 1개 정답 개수 추출 -> 정답 확률 계산
        correct_k = correct[:1].contiguous().view(-1).float().sum(0) # 기존 : correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        pred1 = correct_k.mul_(100.0 / batch_size) # correct_k.mul_ : shape(1)
        # print(f"top1 accuracy : {pred1}")

        return pred1, target, pred

# ==============================================================
# 3. Main문
# ==============================================================
def main():
    # --------------------------------------------------------------
    # 1) 변수 불러오기 + 잘라내기
    # --------------------------------------------------------------
    global args
    args = parser.parse_args()

    # --------------------------------------------------------------
    # 2) Seed 정의 ★★ 필요할까???
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
    warnings.filterwarnings('ignore') # {1} Bang Env

    # --------------------------------------------------------------
    # 3) Model 선언
    # --------------------------------------------------------------
    model = customized_models.__dict__["mobilenetv2"](width_mult=args.width_mult)

    # --------------------------------------------------------------
    # 4) Model to Cuda ★★ cuda 작동될까???
    # --------------------------------------------------------------
    model = torch.nn.DataParallel(model).cuda()

    # --------------------------------------------------------------
    # 5) Loss Function + Optimizer 정의 ★★ cuda 작동될까???
    # --------------------------------------------------------------
    criterion = nn.CrossEntropyLoss().cuda()

    # --------------------------------------------------------------
    # 6) Cudnn 성능 최적화 ★★ cudnn 작동될까???
    # --------------------------------------------------------------
    cudnn.benchmark = True

    # --------------------------------------------------------------
    # 7) DataLoader
    # --------------------------------------------------------------
    # (1) Data 불러오기
    val_loader, val_loader_len, val_files_name = get_pytorch_val_loader(args.data_path, args.batch_size, workers=args.workers, input_size=args.input_size)

    # --------------------------------------------------------------
    # 8) Evaluate : Pretrained Weight 불러오기 -> 평가
    # --------------------------------------------------------------
    from collections import OrderedDict # ★★ OrderedDict 작동될까???
    # (1) StateDict 불러오기
    # # 1] Weight 파일 존재하는 경우 : 불러온다
    if os.path.isfile(args.pretrained_model_path):   # ★★ os 작동될까???
        # print("=> loading pretrained weight '{}'".format(args.pretrained_model_path))
        source_state = torch.load(args.pretrained_model_path) # ★★ load 작동될까???
        # [1] .pt or .pth인 경우 모델 불러오기
        if args.pretrained_model_path[-4:] != ".tar":
            target_state = OrderedDict()
            for k, v in source_state.items():
                if k[:7] != 'module.':
                    k = 'module.' + k
                target_state[k] = v
        # [2] .tar인 경우 모델 불러오기
        else:
            target_state = source_state['state_dict']
        # ==============================================================
        model.load_state_dict(target_state) # ★★ 작동될까???

    # (2) Evaluate
    prec1 = validate(val_loader, val_loader_len, model, criterion, val_files_name)    # {1} Bang Env
    """
    # ==============================================================
    # {1} Bang Env - Print
    # ==============================================================
    print((
        f'Val Loss {val_loss:.04f} | '
        + f'Val Accuracy {prec1:.04f} | '
    ))
    # ==============================================================
    """
    return

# --------------------------------------------------------------
# 8) Evaluate : Pretrained Weight 불러오기 -> 평가
    # (2) Evaluate
# --------------------------------------------------------------
def validate(val_loader, val_loader_len, model, criterion, val_files_name):
    # 1] Meter 변수 생성
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()

    # 2] Evaluate model
    model.eval()

    # 3] Evaluate 계산 + 업데이트 (Learning Rate / 소요 시간 / Output / Loss / Accuracy / Gradient Update)
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # [1] Data Loading 소요시간 측정
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        """
        # ==============================================================
        # {1} Bang Env - 배치 사이즈만큼 돌면서, 파일명 출력
        # ==============================================================
        for j in range(input.size()[0]):
            print("val_files_name : ", val_files_name[i*args.batch_size + j])
        # print(f"input.size() : {input.size()}")
        # ==============================================================
        """

        # [2] Output 확률값 추출 + Loss 계산 (Evaludate이므로 Gradient Update 하지 않음)
        with torch.no_grad():
            output = model(input) # output : shape(64, 1000)

        # [3] Accuracy + Loss 종합적 계산
        prec1, target, pred = accuracy(output, target)
        top1.update(prec1.item(), input.size(0))

        # [4] Batch Data 당 학습 소요시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

        # [5] 결과 시각화
        # for idx, i in enumerate(input):
        #     # 1]] Plt Title
        #     plt.title(f"prec1 : {int(prec1)} | " +
        #               f"target : {int(target[idx].cpu())} | " +
        #               f"pred : {int(pred)} | "
        #               )
        #     # 2]] Plt Subplot Split
        #     plt.subplot(int(len(input) ** 0.5) + 1, int(len(input) ** 0.5), idx + 1)
        #     # 3]] Subplot에 Image 저장
        #     plt.imshow(tensor_to_pltimg(input[idx].cpu()))
        #     # 4]] Plt Figure Axis
        #     plt.axis("off")
        plt.title(f"prec1 : {int(prec1)} | " +
                f"target : {int(target.cpu())} | " +
                  f"pred : {int(pred)} | "
                  )
        plt.imshow(tensor_to_pltimg(input[0].cpu()))
        plt.axis("off")
        plt.show()

    # 4] 평균 Loss / Accuracy 반환
    return top1.avg

# 1. Main문
if __name__ == '__main__':
    main()