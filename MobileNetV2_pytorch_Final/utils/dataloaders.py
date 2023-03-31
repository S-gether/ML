import os
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ==============================================================
# {1} Bang Env - Image Check / Profiler
# ==============================================================
# 0) 라이브러리 불러오기
import time
from PIL import ImageFile

from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

from imagenet import AverageMeter
# ==============================================================

# ==============================================================
# {1} Bang Env - Image Change
# ==============================================================
# 1) OSError: broken data stream when reading image file 에러 해결법
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 2) PIL <-> Numpy <-> Tensor 함수 정의
def pil_to_tensor(pil_image):
    # PIL: [width, height]
    # -> NumPy: [width, height, channel]
    # -> Tensor: [channel, width, height]
    return torch.as_tensor(np.asarray(pil_image)).permute(2,0,1)
def tensor_to_pil(tensor_image):
    return to_pil_image(tensor_image)
def tensor_to_pltimg(tensor_image):
    return tensor_image.permute(1,2,0).numpy()

# 3) Visualization
# (1) Plt Figure Size
plt.figure(figsize=(15, 15))
# ==============================================================

# 4] val dataset 최종 불러오기(batch size / shuffle / num_workers / worker_init_fn / pin_memory / sampler / collate_fn)
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )

    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets

# --------------------------------------------------------------
# 7) DataLoader
    # (1) Data 불러오기
        # 6] Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data) + DataLoader 길이 + all_files 불러오기
# --------------------------------------------------------------
class PrefetchedWrapper(object):
    # [2] Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data)
    def prefetched_loader(loader):
        # 1]] 변수 생성 (mean / std)
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)   # ★★ 작동될까???
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)    # ★★ 작동될까???

        # 2]] GPU 최적화 (Stream 직렬화)
        stream = torch.cuda.Stream() # ★★ torch.cuda.Stream() 작동될까???

        # 3]] Dataset 정규화
        first = True

        """
        # ==============================================================
        # {1} Bang Env - Load Time Check
        # ==============================================================
        load_time = AverageMeter()
        end = time.time()
        # ==============================================================
        """
        # 4]] Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data)
        for idx, (next_input, next_target) in enumerate(loader):
            with torch.cuda.stream(stream): # ★★ torch.cuda.Stream() 작동될까???
                # 1. Batch 단위 Input Data to cuda
                next_input = next_input.cuda(non_blocking=True) # next_input : shape(64, 3, 224, 224) # ★★ cuda 작동될까???

                # 2. Batch 단위 Target Data to cuda
                next_target = next_target.cuda(non_blocking=True) # next_target : shape(64) # ★★ cuda 작동될까???

                # 3. Batch 단위 Input Data 정규화
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            # 4. Dataloader에서 Batch 단위로 Data 순회하면서 불러오기 (Input Data, Target Data)
            if not first:
                yield input, target
            else:
                first = False
            torch.cuda.current_stream().wait_stream(stream) # 역할 ???!!!
            input = next_input
            target = next_target

            """
            # ==============================================================
            # {1} Bang Env - Load Time Check
            # ==============================================================
            load_time.update(time.time() - end)
            print((
                    f'load_count {idx+1}/{len(loader)} | '
                    + f'load_time {load_time.avg}'
            ))
            load_time.reset()
            # ==============================================================
            """
        yield input, target

    # 1] DataLoader 초기화
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    # 2] DataLoader 반복
    def __iter__(self):
        # [1] Epoch 증가
        self.epoch += 1

        # [2] Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data)
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

# --------------------------------------------------------------
# 7) DataLoader
    # (1) Data 불러오기
# --------------------------------------------------------------
def get_pytorch_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None, input_size=224):
    # 1] val dataset 경로 불러오기
    valdir = os.path.join(data_path, 'val') # ★★ 작동될까???

    # 2] val dataset 이미지 불러오기 + Augmentation
    val_dataset = datasets.ImageFolder( # 구조!!! ★★ 작동될까???
            valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                ]))
    # print("val_dataset : ", val_dataset)

    # 3] Sampler 설정
    val_sampler = None

    # 4] val dataset 최종 불러오기(batch size / shuffle / num_workers / worker_init_fn / pin_memory / sampler / collate_fn)
    val_loader = torch.utils.data.DataLoader( # 구조!!! ★★ dataloader 작동될까???
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=fast_collate)

    # 5] val files 이름 불러오기
    val_files_name, _ = map(list, zip(*val_loader.dataset.samples)) # ★★ val_loader.dataset.samples 작동될까???

    # 6] Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data) + DataLoader 길이 + all_files 불러오기
    return PrefetchedWrapper(val_loader), len(val_loader), val_files_name