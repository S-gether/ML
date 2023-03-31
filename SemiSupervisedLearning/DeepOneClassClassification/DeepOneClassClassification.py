# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# 1) 기본 라이브러리
import numpy as np
import easydict
from PIL import Image

# 2) 딥러닝 라이브러리
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.onnx

# ==============================================================
# 0. 변수 정의
# ==============================================================
args = easydict.EasyDict({
        'pretrain':True,
        'num_epochs':50,
        'num_epochs_ae':50,
        'lr':1e-3,
        'lr_ae':1e-3,
        'lr_milestones':[50],
        'weight_decay':5e-7,
        'weight_decay_ae':5e-3,
        'batch_size':1024,
        'latent_dim':32,
        'normal_class':0,
        })

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------------------
# 1) 데이터 로더 불러오기
# (3-3) 데이터 불러오기 (+ 증강) : Preprocessing을 포함한 dataloader를 구성
# --------------------------------------------------------------
class MNIST_loader(data.Dataset):
    # 1] 초기 변수 세팅
    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    # 2] 데이터셋 불러오기 (Data / Target) + 증강
    def __getitem__(self, index):
        # [1] 특정 index 데이터 불러오기
        x = self.data[index]
        y = self.target[index]
        # [2] 데이터 증강
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y

    # 3] 데이터셋 개수 불러오기
    def __len__(self):
        return len(self.data)

# --------------------------------------------------------------
# 1. Main문
    # 1) 데이터 로더 불러오기 (+ 증강)
# --------------------------------------------------------------
def get_mnist(args, data_dir='../data/'):
    # (1) 데이터 증강 정의
    # 1] min_max # 이건 왜있음???
    min_max = [(-0.8826567065619495, 9.001545489292527),
               (-0.6661464580883915, 20.108062262467364),
               (-0.7820454743183202, 11.665100841080346),
               (-0.7645772083211267, 12.895051191467457),
               (-0.7253923114302238, 12.683235701611533),
               (-0.7698501867861425, 13.103278415430502),
               (-0.778418217980696, 10.457837397569108),
               (-0.7129780970522351, 12.057777597673047),
               (-0.8280402650205075, 10.581538445782988),
               (-0.7369959242164307, 10.697039838804978)]
    # 2] contrast_normalization : X / mean(abs(X-X_mean))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x)),
                                    transforms.Normalize([min_max[args.normal_class][0]], [min_max[args.normal_class][1] - min_max[args.normal_class][0]])])

    # (2-1) 데이터셋 불러오기
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    # (3-1) Train 데이터 Split
    x_train = train.data
    y_train = train.targets

    # (3-2) 데이터 전처리 (Normal Class인 경우만 불러오기)
    x_train = x_train[np.where(y_train == args.normal_class)]
    y_train = y_train[np.where(y_train == args.normal_class)]

    # (3-3) 데이터 불러오기 (+증강)
    data_train = MNIST_loader(x_train, y_train, transform)

    # (3-4) 데이터 로더 불러오기
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

    # (3-1) Test 데이터 Split
    x_test = test.data
    y_test = test.targets

    # (3-2) 데이터 전처리 (Normal Class인 경우만 0으로 변환, 아닌 경우 1로 변환)
    y_test = np.where(y_test == args.normal_class, 0, 1) # Normal class인 경우 0으로 바꾸고, 나머지는 1로 변환 (정상 vs 비정상 class)

    # (3-3) 데이터 불러오기 (+증강)
    data_test = MNIST_loader(x_test, y_test, transform)

    # (3-4) 데이터 로더 불러오기
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test

# --------------------------------------------------------------
# 1. Main문
    # 1) 데이터 로더 불러오기
        # (1) 데이터 증강
            # 1] contrast_normalization : X / mean(abs(X-X_mean)) : Apply global contrast normalization to tensor
# --------------------------------------------------------------
def global_contrast_normalization(x): # x shape : (1, 28, 28)
    # print(f"x : {x}")
    """
    tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000], ...]])
    """
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    # print(f"mean : {mean}") # tensor(0.1650)
    x -= mean
    # print(f"x : {x}")
    """
    tensor([[[-0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650,
      -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650,
      -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650,
      -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650],
     [-0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650,
      -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650,
      -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650,
      -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650, -0.1650], ...]])
    """
    x_scale = torch.mean(torch.abs(x))
    # print(f"x_scale : {x_scale}") # x_scale : 0.2693599760532379
    x /= x_scale
    return x

# --------------------------------------------------------------
# 1. Main문
    # 2) 모델 객체 선언
        # (3) 모델 학습
            # 1] 모델 불러오기
# --------------------------------------------------------------
class DeepSVDD_network(nn.Module):
    # [1] 초기 변수 세팅
    def __init__(self, z_dim=32):
        # 1]] nn.Module 상속
        super(DeepSVDD_network, self).__init__()

        # 2]] Layer 생성
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

    # [2] 순전파
    def forward(self, x):
        # [Layer] Conv -> [Regularization] Batch Normalization -> [Activate Function] Leaky Relu -> [Pooling] MaxPooling -> ...
        x = self.conv1(x) # x shape : (1024, 8, 28, 28)
        x = self.pool(F.leaky_relu(self.bn1(x))) # x shape : (1024, 8, 14, 14)
        x = self.conv2(x) # x shape : (1024, 4, 14, 14)
        x = self.pool(F.leaky_relu(self.bn2(x))) # x shape : (1024, 4, 7, 7)
        x = x.view(x.size(0), -1) # x shape : (1024, 196)
        return self.fc1(x) # x shape : (196, 32)

# --------------------------------------------------------------
# 1. Main문
    # 2) 모델 객체 선언
        # (2) 사전 학습
            # 1] Pretrain AutoEncoder
# --------------------------------------------------------------
class pretrain_autoencoder(nn.Module):
    # [1] 초기 설정
    def __init__(self, z_dim=32):
        # 1]] nn.Module 상속
        super(pretrain_autoencoder, self).__init__()

        # 2]] Layer 생성
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    # [2] Encoder
    def encoder(self, x): # x shape : (1024, 1, 28, 28)
        # [Layer] Conv -> [Regularization] Batch Normalization -> [Activate Function] Leaky Relu -> [Pooling] MaxPooling -> ...
        """
        tensor([[[[0.0276, 0.0276, 0.0276,  ..., 0.0276, 0.0276, 0.0276],
          [0.0276, 0.0276, 0.0276,  ..., 0.0276, 0.0276, 0.0276],
          [0.0276, 0.0276, 0.0276,  ..., 0.0276, 0.0276, 0.0276],
          ...,
          [0.0276, 0.0276, 0.0276,  ..., 0.0276, 0.0276, 0.0276],
          [0.0276, 0.0276, 0.0276,  ..., 0.0276, 0.0276, 0.0276],
          [0.0276, 0.0276, 0.0276,  ..., 0.0276, 0.0276, 0.0276]]],
        ...,
        [[[0.0277, 0.0277, 0.0277,  ..., 0.0277, 0.0277, 0.0277],
          [0.0277, 0.0277, 0.0277,  ..., 0.0277, 0.0277, 0.0277],
          [0.0277, 0.0277, 0.0277,  ..., 0.0277, 0.0277, 0.0277],
          ...,
          [0.0277, 0.0277, 0.0277,  ..., 0.0277, 0.0277, 0.0277],
          [0.0277, 0.0277, 0.0277,  ..., 0.0277, 0.0277, 0.0277],
          [0.0277, 0.0277, 0.0277,  ..., 0.0277, 0.0277, 0.0277]]]],
        device='cuda:0')
        """
        x = self.conv1(x) # x shape : (1024, 8, 28, 28)
        """
        tensor([[[[ 8.4724e-04, -5.8248e-04, -1.5504e-03,  ..., -1.5504e-03,
           -1.3454e-03, -2.3005e-03],
          [ 2.0023e-03,  7.9786e-04, -7.5894e-04,  ..., -7.5894e-04,
           -7.6778e-04, -1.5384e-03],
          [ 1.2658e-03,  7.4467e-05, -2.5211e-03,  ..., -2.5211e-03,
           -2.6225e-03, -3.5191e-03],
          ...,
          [ 1.2658e-03,  7.4467e-05, -2.5211e-03,  ..., -2.5211e-03,
           -2.6225e-03, -3.5191e-03],
          [ 6.8929e-04, -5.8720e-04, -2.4498e-03,  ..., -2.4498e-03,
           -2.8380e-03, -3.4151e-03],
          [ 1.4544e-03,  1.0358e-03, -9.5891e-04,  ..., -9.5891e-04,
           -1.7916e-03, -2.3276e-03]],
         ...,
         [[ 1.5558e-03,  1.0378e-03, -2.0889e-04,  ..., -2.0889e-04,
           -5.0021e-04,  4.1879e-04],
          [ 1.3592e-03,  3.0450e-04, -8.4818e-04,  ..., -8.4818e-04,
           -8.0985e-04,  4.6607e-04],
          [-1.2935e-04, -1.5629e-03, -2.6455e-03,  ..., -2.6455e-03,
           -2.5459e-03, -2.4398e-04],
          ...,
          [-1.2935e-04, -1.5629e-03, -2.6455e-03,  ..., -2.6455e-03,
           -2.5459e-03, -2.4398e-04],
          [-1.0068e-03, -1.8687e-03, -2.3457e-03,  ..., -2.3457e-03,
           -1.5685e-03,  5.5790e-04],
          [-7.0794e-04, -1.3085e-03, -1.2051e-03,  ..., -1.2051e-03,
           -7.0351e-04,  6.1333e-04]]]], device='cuda:0',
       grad_fn=<ConvolutionBackward0>)
        """
        x = self.pool(F.leaky_relu(self.bn1(x))) # x shape : (1024, 8, 14, 14)
        x = self.conv2(x) # x shape : (1024, 4, 14, 14)
        x = self.pool(F.leaky_relu(self.bn2(x)))# x shape : (1024, 4, 7, 7)
        x = x.view(x.size(0), -1)# x shape : (1024, 196)
        return self.fc1(x) # x shape : (196, 32)

    # [3] Decoder
    def decoder(self, x): # x shape : (1024, 32)
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4) # x shape : (1024, 2, 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2) # x shape : (1024, 2, 8, 8)
        x = self.deconv1(x) # x shape : (1024, 4, 8, 8)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2) # x shape : (1024, 4, 16, 16)
        x = self.deconv2(x) # x shape : (1024, 8, 14, 14)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2) # x shape : (1024, 8, 28, 28)
        x = self.deconv3(x) # x shape : (1024, 1, 28, 28)
        return torch.sigmoid(x) # x shape : (1024, 1, 28, 28)

    # [4] 순전파
    def forward(self, x):
        z = self.encoder(x) # x shape : (196, 32)
        x_hat = self.decoder(z) # x shape : (1024, 1, 28, 28)
        return x_hat

# --------------------------------------------------------------
# 1. Main문
    # 2) 모델 객체 선언
# --------------------------------------------------------------
class TrainerDeepSVDD:
    # (1) 초기 변수 세팅
    def __init__(self, args, data_loader, device):
        self.args = args
        self.train_loader = data_loader
        self.device = device

    # (2) Pretrain AutoEncoder 모델 사전 학습 (역할 : DeepSVDD 모델에서 사용할 가중치를 학습시키는 AutoEncoder 학습 단계 / 대응 역할 : Kernel Mapping / 목표 : Input을 가장 잘 복원하는 각 노드별 Kernel Mapping된 C 구하기)
    def pretrain(self):
        # 1] Pretrain AutoEncoder 모델 불러오기
        ae = pretrain_autoencoder(self.args.latent_dim).to(self.device)

        # 2] Pretrain AutoEncoder 모델 가중치 정규화
        ae.apply(weights_init_normal) # net.apply : 해당 Module의 모든 Sub-Module에 인수받은 함수를 적용

        # 3] Optimizer Define
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae, weight_decay=self.args.weight_decay_ae)
        # 4] Scheduler Define
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        # 5] Train Pretrain AutoEncoder 모델
        ae.train()

        # 6] Epoch 순회 (Update Train / Print)
        for epoch in range(self.args.num_epochs_ae):
            # [1] 데이터 로더 불러오기 + 학습
            total_loss = 0
            for x, _ in self.train_loader:
                # 1]] 데이터 자료형 변환
                x = x.float().to(self.device)

                # 2]] Gradient 초기화
                optimizer.zero_grad()
                # 3]] Pretrain AutoEncoder Output 추출 (Train이므로 Gradient Update 필요)
                x_hat = ae(x)
                # 4]] Loss 계산 + Step # !!!
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                # 5]] Total Loss 계산
                total_loss += reconst_loss.item()
            scheduler.step()

            # [2] 시각화
            print(f"[Pretraining Autoencoder] Epoch {epoch} | " + f"Mean Loss {total_loss / len(self.train_loader):.3f}")

        # 7] 원 중심 & 모델 가중치 (Encoder) 전달 + 저장 (학습된 AutoEncoder 가중치를 DeepSVDD모델에 Initialize해주는 함수)
        self.save_weights_for_DeepSVDD(ae, self.train_loader)

    # --------------------------------------------------------------
    # 1. Main문
        # 2) 모델 객체 선언
            # (2) 모델 사전 학습
                # 7] 원 중심 & 모델 가중치 (Encoder) 전달 + 저장 (역할 : 학습된 AutoEncoder 가중치를 DeepSVDD모델에 Initialize해주는 함수)
    # --------------------------------------------------------------
    def save_weights_for_DeepSVDD(self, model, dataloader):
        # [1] 원 중심 세팅
        c = self.set_c(model, dataloader)

        # [2] DeepSVDD_network 모델 불러오기
        net = DeepSVDD_network(self.args.latent_dim).to(self.device)

        # [3] 모델 가중치 전달 : pretrain_autoencoder 모델 가중치 (Encoder) -> DeepSVDD_network 모델 가중치
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)

        # [4] 원 중심 + DeepSVDD_network 모델 가중치 저장
        torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': net.state_dict()}, './weights/pretrained_parameters.pth')

    # --------------------------------------------------------------
    # 1. Main문
        # 2) 모델 객체 선언
            # (2) 모델 사전 학습
                # 7] 모델 가중치 전달 : pretrain_autoencoder 모델 가중치 (Encoder) -> DeepSVDD_network 모델 가중치
                    # [1] 원 중심 세팅 (역할 : Initializing The Center for The Hypersphere)
    # --------------------------------------------------------------
    def set_c(self, model, dataloader, eps=0.1):
        # 1]] 모델 평가
        model.eval()

        # 2]] 데이터 로더 불러오기 + 예측
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader: # x : (1024, 1, 28, 28)
                # [[1]] 데이터 자료형 변환
                x = x.float().to(self.device) # x : (1024, 1, 28, 28)
                # [[2]] Pretrain AutoEncoder Output 추출 (Train이므로 Gradient Update 필요)
                z = model.encoder(x) # x : (1024, 32)
                # [[3]] Output 역전파 끊기 + list 쌓기
                z_.append(z.detach()) # (5923, 32)

        # 3]] Pretrain AutoEncoder Output 연결
        z_ = torch.cat(z_) # (5923, 32)
        # 4]] Pretrain AutoEncoder Output 클래스별 평균 계산
        c = torch.mean(z_, dim=0) # (32)
        # print(f"c : {c}")
        """
        tensor([-0.4863,  0.5074,  0.7331,  0.7768,  0.4090, -0.7834,  1.2013, -0.5883,
        -0.7117,  1.1654, -2.1758,  1.1147, -0.7257, -0.4698, -0.9613,  0.7392,
         0.7901,  0.6481, -0.3062,  0.4030,  0.7793, -1.4222, -0.6568, -1.0333,
         0.6771, -0.9004, -0.8838,  0.9348,  1.1437,  0.0387,  1.3928,  0.3372],
       device='cuda:0')
        """
        # 5]]
        c[(abs(c) < eps) & (c < 0)] = -eps
        # print(f"(abs(c) < eps) & (c < 0) : {(abs(c) < eps) & (c < 0)}")
        """
        tensor([False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False], device='cuda:0')
        """
        # print(f"c : {c}")
        """
        tensor([-0.4863,  0.5074,  0.7331,  0.7768,  0.4090, -0.7834,  1.2013, -0.5883,
        -0.7117,  1.1654, -2.1758,  1.1147, -0.7257, -0.4698, -0.9613,  0.7392,
         0.7901,  0.6481, -0.3062,  0.4030,  0.7793, -1.4222, -0.6568, -1.0333,
         0.6771, -0.9004, -0.8838,  0.9348,  1.1437,  0.0387,  1.3928,  0.3372],
       device='cuda:0')
        """
        c[(abs(c) < eps) & (c > 0)] = eps
        # print(f"(abs(c) < eps) & (c > 0) : {(abs(c) < eps) & (c > 0)}")
        """
        tensor([False, False, False, False, False, False, False, False, False, False,
                False, False, False, False, False, False, False, False, False, False,
                False, False, False, False, False, False, False, False, False, False,
                False, False], device='cuda:0')
        """
        # print(f"c : {c}")
        """
        tensor([-0.4863,  0.5074,  0.7331,  0.7768,  0.4090, -0.7834,  1.2013, -0.5883,
        -0.7117,  1.1654, -2.1758,  1.1147, -0.7257, -0.4698, -0.9613,  0.7392,
         0.7901,  0.6481, -0.3062,  0.4030,  0.7793, -1.4222, -0.6568, -1.0333,
         0.6771, -0.9004, -0.8838,  0.9348,  1.1437,  0.1000,  1.3928,  0.3372],
       device='cuda:0')
        """
        return c

    # (3) DeepSVDD_network 모델 학습 (역할 : Deep SVDD model 학습 / 대응 역할 : Classifier / 목표 : 각 노드별 Classifier와 C 사이 거리 최소화 훈련)
    def train(self):
        # 1] DeepSVDD_network 모델 불러오기
        net = DeepSVDD_network().to(self.device)

        # 2] DeepSVDD_network 모델 가중치 + 원 중심 초기화
        # [1] Pretrain AutoEncoder 사전 학습 모델 가중치 가져오는 경우
        if self.args.pretrain == True:
            # 1]] DeepSVDD_network 모델 가중치 + 원 중심 불러오기
            state_dict = torch.load('./weights/pretrained_parameters.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        # [2] Pretrain AutoEncoder 사전 학습 모델 가중치 가져오지 않는 경우
        else:
            # 1]] 가중치 정규화
            net.apply(weights_init_normal) # net.apply : 해당 Module의 모든 Sub-Module에 인수받은 함수를 적용
            # 2]] 원의 중심 초기화 by 정규분포 난수
            c = torch.randn(self.args.latent_dim).to(self.device)

        # 3] Optimizer Define
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # 4] Scheduler Define
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        # 5] Train Model
        net.train()

        # 6] Epoch 순회 (Update Train / Print)
        for epoch in range(self.args.num_epochs):
            # [1] 데이터 로더 불러오기 + 학습
            total_loss = 0
            for x, _ in self.train_loader:
                # 1]] 데이터 자료형 변환
                x = x.float().to(self.device)

                # 2]] Gradient 초기화
                optimizer.zero_grad()
                # 3]] DeepSVDD_network Output 추출 (Train이므로 Gradient Update 필요)
                z = net(x)
                # 4]] Loss 계산 + Step
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1)) # torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                # 5]] Total Loss 계산
                total_loss += loss.item()
            scheduler.step()

            # [2] 시각화
            print(f"[DeepSVDD_network] Epoch {epoch} | " + f"Mean Loss {total_loss / len(self.train_loader):.3f}")

        # 7] Model / 원의 중심 저장
        self.net = net
        self.c = c
        return self.net, self.c


# --------------------------------------------------------------
# 1. Main문
    # 3-3) 모델 평가 (역할 : Testing the Deep SVDD model / 대응 역할 : Classifier / 목표 : 각 노드별 Classifier와 C 사이 거리 최소화 확인)
# --------------------------------------------------------------
def DeepSVDD_network_eval(net, c, dataloader, device):
    scores = []
    labels = []

    # 1]] 모델 평가
    net.eval()

    # 2]] 데이터 로더 불러오기 + 예측
    with torch.no_grad():
        for x, y in dataloader:
            # [[1]] 데이터 자료형 변환
            x = x.float().to(device)
            # [[2]] DeepSVDD_network Output 추출 (Train이므로 Gradient Update 필요)
            z = net(x)
            # [[3]] Loss 계산
            score = torch.sum((z - c) ** 2, dim=1)

            # [[4]] Loss 역전파 끊기 + list 쌓기
            scores.append(score.detach().cpu())
            # [[4]] Label list 쌓기
            labels.append(y.cpu())
    # 3]] Loss 연결 / Label 연결
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()

    # 4]] 시각화
    print(f"[ROC AUC score: {roc_auc_score(labels, scores)*100:.2f}")
    return labels, scores

# --------------------------------------------------------------
# 1. Main문
    # 2) 모델 객체 선언
        # (2) 모델 사전 학습
            # 2] 모델 가중치 정규화
# --------------------------------------------------------------
# --------------------------------------------------------------
# 1. Main문
    # 2) 모델 객체 선언
        # (3) 모델 학습
            # 2] 모델 가중치 + 원 중심 초기화
                # [2] 사전 학습 모델 가중치 가져오지 않는 경우
                    # 1]] 가중치 정규화
# --------------------------------------------------------------
def weights_init_normal(m): # m : Linear(in_features=196, out_features=32, bias=False)
    classname = m.__class__.__name__ # classname : 'Linear'
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # m.weight.data shape : (32, 196)
        """
        m.weight.data : 
        tensor([[ 0.0702,  0.0124, -0.0620,  ...,  0.0126,  0.0574,  0.0068],
        [ 0.0565,  0.0251, -0.0608,  ...,  0.0015,  0.0353,  0.0042],
        [-0.0566, -0.0501,  0.0696,  ..., -0.0199,  0.0353,  0.0343],
        ...,
        [-0.0202, -0.0551,  0.0307,  ..., -0.0041, -0.0019,  0.0007],
        [ 0.0365,  0.0010, -0.0428,  ...,  0.0118,  0.0017,  0.0104],
        [-0.0075, -0.0703, -0.0097,  ...,  0.0590,  0.0431, -0.0133]],
       device='cuda:0')
        """

# ==============================================================
# 1. Main문
# ==============================================================
if __name__ == '__main__':
    # 1) 데이터 로더 불러오기 (+ 증강)
    dataloader_train, dataloader_test = get_mnist(args)

    # torch.onnx.export(pretrain_autoencoder(), torch.empty(1024, 1, 28, 28, dtype = torch.float32), "output.onnx")

    # 2) 모델 객체 선언
    deep_SVDD = TrainerDeepSVDD(args, dataloader_train, device)

    # 3-1) 모델 학습 by 사전 학습된 가중치
    if args.pretrain:
        deep_SVDD.pretrain()

    # 3-2) 모델 학습 by 학습된 가중치
    net, c = deep_SVDD.train()

    # 3-3) 모델 평가
    labels, scores = DeepSVDD_network_eval(net, c, dataloader_test, device)
    print(f"labels {labels} | score {scores:.3f}")