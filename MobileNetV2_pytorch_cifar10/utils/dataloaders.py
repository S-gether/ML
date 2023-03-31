import os
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ==============================================================
# {1} Bang Env - Image Check
# ==============================================================
# 0) 라이브러리 불러오기
import time
from PIL import ImageFile

from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

from utils import AverageMeter

# from imagenet import args

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

DATA_BACKEND_CHOICES = ['pytorch']
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    DATA_BACKEND_CHOICES.append('dali-gpu')
    DATA_BACKEND_CHOICES.append('dali-cpu')
except ImportError:
    print("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = local_rank,
                num_shards = world_size,
                random_shuffle = True)

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                                                    random_aspect_ratio=[0.75, 4./3.],
                                                    random_area=[0.08, 1.0],
                                                    num_attempts=100)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512,
                                                      random_aspect_ratio=[0.75, 4./3.],
                                                      random_area=[0.08, 1.0],
                                                      num_attempts=100)

        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            output_layout = types.NCHW,
                                            crop = (crop, crop),
                                            image_type = types.RGB,
                                            mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                                            std = [0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability = 0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror = rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = local_rank,
                num_shards = world_size,
                random_shuffle = False)

        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.Resize(device = "gpu", resize_shorter = size)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                output_dtype = types.FLOAT,
                output_layout = types.NCHW,
                crop = (crop, crop),
                image_type = types.RGB,
                mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                std = [0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


class DALIWrapper(object):
    def gen_wrapper(dalipipeline):
        for data in dalipipeline:
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline):
        self.dalipipeline = dalipipeline

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.dalipipeline)

def get_dali_train_loader(dali_cpu=False):
    def gdtl(data_path, batch_size, workers=5, _worker_init_fn=None):
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        traindir = os.path.join(data_path, 'train')

        pipe = HybridTrainPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = traindir, crop = 224, dali_cpu=dali_cpu)

        pipe.build()
        test_run = pipe.run()
        train_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size))

        return DALIWrapper(train_loader), int(pipe.epoch_size("Reader") / (world_size * batch_size))

    return gdtl


def get_dali_val_loader():
    def gdvl(data_path, batch_size, workers=5, _worker_init_fn=None):
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        valdir = os.path.join(data_path, 'val')

        pipe = HybridValPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = valdir,
                crop = 224, size = 256)
        pipe.build()
        test_run = pipe.run()
        val_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size), fill_last_batch=False)

        return DALIWrapper(val_loader), int(pipe.epoch_size("Reader") / (world_size * batch_size))
    return gdvl

# [4] train dataset 최종 불러오기 (batch size / shuffle / num_workers / worker_init_fn / pin_memory / sampler / collate_fn)
def fast_collate(batch): # 구조!!!
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
# 10) DataLoader
    # (1) Data 불러오는 형식
        # 1] Data Backend == Pytorch인 경우
            # [5] Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data)
# --------------------------------------------------------------
class PrefetchedWrapper(object):
    # 3. Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data)
    def prefetched_loader(loader):
        # 1) 변수 생성
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # print("mean", mean) # tensor([[[[123.6750]], [[116.2800]], [[103.5300]]]], device='cuda:0')
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # print("std", std) # tensor(tensor([[[[58.3950]], [[57.1200]], [[57.3750]]]], device='cuda:0'))

        # 2) GPU 최적화 (Stream 직렬화)
        stream = torch.cuda.Stream()

        # 3) Dataset 정규화
        first = True
        # ==============================================================
        # {1} Bang Env - Load Time Check
        # ==============================================================
        load_time = AverageMeter()
        end = time.time()
        # ==============================================================
        # (1) Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data)
        for idx, (next_input, next_target) in enumerate(loader):
            with torch.cuda.stream(stream):
                # 1] Batch 단위 Input Data to cuda
                next_input = next_input.cuda(non_blocking=True) # next_input : shape(64, 3, 224, 224)
                """
                tensor([[[[-0.5082, -0.3883, -0.4226,  ...,  0.9303,  0.3823,  0.6392],
                [-0.6281, -0.6965, -0.4397,  ...,  0.8104,  0.5878,  0.2111],
                [-0.5767, -0.1486,  0.0741,  ...,  0.7419,  0.8961,  0.2282],
                ...,
                [-0.3369, -0.3883, -0.3541,  ..., -0.0629, -0.2856, -0.4226],
                [-0.2856, -0.3198, -0.3369,  ..., -0.4568,  0.1083,  0.5364],
                [-0.2342, -0.2513, -0.1657,  ..., -0.2684, -0.3369, -0.3198]],
                
                [[-0.5651, -0.4076, -0.0399,  ...,  1.2731,  0.7479,  1.0280],
                [-0.5301, -0.6352, -0.3550,  ...,  1.1331,  0.8880,  0.5378],
                [-0.4426, -0.0924,  0.1176,  ...,  1.0805,  1.2906,  0.6604],
                ...,
                [-0.1625, -0.2500, -0.2325,  ...,  0.0476, -0.0924, -0.0399],
                [ 0.0826, -0.1099, -0.2325,  ..., -0.0924,  0.4153,  0.9580],
                [ 0.6604,  0.3452,  0.3627,  ...,  0.1176, -0.0749, -0.1450]],
                
                [[-0.7413, -0.5495,  0.0779,  ...,  1.0888,  0.7576,  0.7054],
                [-0.8110, -0.7936, -0.4450,  ...,  1.1934,  1.0191,  0.6705],
                [-0.7936, -0.4275, -0.1312,  ...,  1.0365,  1.3328,  0.7228],
                ...,
                [-0.2358, -0.2881, -0.2184,  ...,  0.0082, -0.2010, -0.2532],
                [ 0.1999, -0.0790, -0.2010,  ..., -0.2184,  0.4962,  1.1411],
                [ 0.9319,  0.6705,  0.5834,  ...,  0.3219, -0.1138, -0.3230]]],
                
                
                [[[-1.4329, -1.7583, -1.6898,  ..., -1.6898, -1.7069, -1.6727],
                [-1.6042, -1.7754, -1.7583,  ..., -1.5357, -1.6384, -1.7412],
                [-1.6727, -1.9124, -1.8782,  ..., -1.4329, -1.4329, -1.4158],
                ...,
                [-0.2513, -0.1486, -0.2342,  ..., -1.4329, -1.4158, -1.3473],
                [-0.2684, -0.1999, -0.2684,  ..., -1.6213, -1.6384, -1.5699],
                [-0.3198, -0.3027, -0.3198,  ..., -1.6213, -1.6042, -1.6384]],
                
                [[-1.1078, -1.4055, -1.4230,  ..., -1.1954, -1.2304, -1.2129],
                [-1.2829, -1.4755, -1.5280,  ..., -1.3354, -1.4230, -1.5280],
                [-1.4580, -1.7381, -1.7731,  ..., -1.2479, -1.2129, -1.2129],
                ...,
                [-0.0749,  0.0301, -0.0224,  ..., -1.2829, -1.2479, -1.0903],
                [-0.1275, -0.0749, -0.0924,  ..., -1.2479, -1.2829, -1.2829],
                [-0.1625, -0.1800, -0.1625,  ..., -1.2829, -1.3004, -1.3704]],
                
                [[-0.4973, -0.7936, -0.8981,  ..., -1.2990, -1.3164, -1.2641],
                [-0.8633, -1.0201, -1.1073,  ..., -1.1944, -1.2990, -1.3861],
                [-1.1073, -1.3861, -1.4210,  ..., -1.0201, -1.0376, -1.0550],
                ...,
                [-0.2707, -0.1835, -0.2881,  ..., -0.8458, -0.8110, -0.7587],
                [-0.2881, -0.2532, -0.2707,  ..., -0.7587, -0.7936, -0.8284],
                [-0.3230, -0.3404, -0.3055,  ..., -0.8110, -0.8284, -0.9156]]],
                
                
                [[[-1.9467, -0.9877, -0.5596,  ..., -0.0287,  0.0569, -0.6281],
                [-1.8439, -1.5185, -1.8953,  ...,  0.3138,  0.1597, -0.2856],
                [-1.7925, -1.9809, -2.0494,  ..., -0.8507, -1.1075, -0.3198],
                ...,
                [ 1.1187,  1.4783,  1.7180,  ..., -1.9809, -1.9638, -1.9809],
                [ 1.3927,  1.7180,  1.7009,  ..., -1.9980, -1.9809, -1.9980],
                [ 1.4783,  1.4783,  0.6221,  ..., -1.9638, -1.9809, -1.9809]],
                
                [[-1.3004, -0.6702, -0.3550,  ...,  0.7304,  0.8354,  0.1001],
                [-1.2129, -1.1604, -1.7906,  ...,  1.1155,  0.9055,  0.5028],
                [-1.2304, -1.7556, -1.9482,  ..., -0.0749, -0.4776,  0.3627],
                ...,
                [ 0.7829,  1.5007,  1.6232,  ..., -1.8081, -1.7906, -1.8081],
                [ 1.1155,  1.7108,  1.7108,  ..., -1.8256, -1.8081, -1.8256],
                [ 1.1681,  1.1506,  0.4853,  ..., -1.7906, -1.8081, -1.8081]],
                
                [[-1.6824, -1.1944, -0.9156,  ..., -0.5321, -0.5321, -0.8458],
                [-1.7173, -1.4210, -1.6127,  ..., -0.0964, -0.2532, -0.4798],
                [-1.6650, -1.6824, -1.6824,  ..., -1.0376, -1.0201, -0.4101],
                ...,
                [ 0.1128,  0.6879,  0.9319,  ..., -1.5256, -1.5081, -1.5081],
                [ 0.8099,  1.1585,  0.9494,  ..., -1.5430, -1.5256, -1.5256],
                [ 0.6879,  0.5311, -0.0615,  ..., -1.5430, -1.5430, -1.5081]]],
                
                
                ...,
                
                
                [[[ 1.3584,  1.3584,  1.3755,  ...,  0.5193,  0.5364,  0.5536],
                [ 1.3413,  1.3584,  1.3755,  ...,  0.5707,  0.5193,  0.5193],
                [ 1.3584,  1.3927,  1.4098,  ...,  0.5364,  0.4679,  0.4679],
                ...,
                [ 0.9474,  0.9303,  0.9646,  ...,  0.4679,  0.4337,  0.3481],
                [ 0.9474,  0.9646,  0.9303,  ...,  0.3652,  0.3652,  0.3823],
                [ 0.8961,  0.9132,  0.9646,  ...,  0.3994,  0.3309,  0.3823]],
                
                [[ 1.6758,  1.6408,  1.6583,  ...,  1.3957,  1.3782,  1.3782],
                [ 1.6408,  1.6583,  1.6933,  ...,  1.3782,  1.3782,  1.3606],
                [ 1.6057,  1.6758,  1.7108,  ...,  1.3782,  1.3782,  1.3431],
                ...,
                [ 1.5357,  1.5357,  1.5357,  ...,  1.3081,  1.2906,  1.2906],
                [ 1.5182,  1.5357,  1.5532,  ...,  1.3081,  1.2906,  1.2906],
                [ 1.5357,  1.5182,  1.5182,  ...,  1.2731,  1.3081,  1.2731]],
                
                [[ 2.0125,  2.0125,  1.9777,  ..., -0.4798, -0.5321, -0.5147],
                [ 1.9777,  1.9951,  2.0125,  ..., -0.4624, -0.5147, -0.5321],
                [ 1.9428,  2.0474,  2.0648,  ..., -0.4624, -0.5495, -0.5321],
                ...,
                [ 0.5834,  0.4962,  0.4614,  ..., -0.7587, -0.7936, -0.8110],
                [ 0.5659,  0.5136,  0.5136,  ..., -0.7936, -0.7936, -0.7761],
                [ 0.4962,  0.5311,  0.5659,  ..., -0.7587, -0.8110, -0.7936]]],
                
                
                [[[ 0.0912,  0.0741,  0.0398,  ..., -1.5185, -1.5357, -1.4329],
                [ 0.0912,  0.0741,  0.0056,  ..., -1.5014, -1.5528, -1.4500],
                [ 0.1426,  0.0912,  0.0227,  ..., -1.5528, -1.5699, -1.4500],
                ...,
                [-1.3987, -1.4500, -1.4500,  ...,  1.1700,  1.1358,  1.1358],
                [-1.3473, -1.3987, -1.3987,  ...,  1.1187,  1.1187,  1.1015],
                [-1.3473, -1.3473, -1.3815,  ...,  1.1872,  1.1700,  1.1529]],
                
                [[-0.4601, -0.4776, -0.4951,  ..., -1.5630, -1.5280, -1.4055],
                [-0.4601, -0.4951, -0.4951,  ..., -1.5105, -1.5455, -1.4055],
                [-0.4251, -0.4776, -0.4951,  ..., -1.5280, -1.5630, -1.4580],
                ...,
                [-1.6331, -1.6681, -1.6856,  ..., -1.0553, -1.0378, -1.0028],
                [-1.5980, -1.6331, -1.6506,  ..., -1.0203, -1.0203, -1.0028],
                [-1.5980, -1.6155, -1.6331,  ..., -0.9153, -0.9503, -0.9853]],
                
                [[-0.8284, -0.7587, -0.7238,  ..., -1.3513, -1.3339, -1.2641],
                [-0.7761, -0.7587, -0.7413,  ..., -1.2641, -1.2990, -1.2293],
                [-0.6367, -0.6890, -0.6890,  ..., -1.3339, -1.3861, -1.3164],
                ...,
                [-1.3861, -1.4036, -1.4210,  ..., -1.4559, -1.4210, -1.3861],
                [-1.3513, -1.3861, -1.4036,  ..., -1.4559, -1.4384, -1.4559],
                [-1.3339, -1.3513, -1.3861,  ..., -1.3861, -1.3861, -1.3861]]],
                
                
                [[[-0.4568, -0.4226, -0.4568,  ..., -0.7479, -0.7822, -1.0390],
                [-0.3541, -0.4054, -0.4054,  ..., -0.7479, -0.8164, -0.9534],
                [-0.3883, -0.4226, -0.3883,  ..., -0.7822, -0.9192, -0.8678],
                ...,
                [-1.9467, -1.9638, -1.9124,  ..., -2.0665, -2.0323, -2.0837],
                [-1.7412, -1.7412, -1.7925,  ..., -2.1008, -2.0665, -2.0323],
                [-1.7240, -1.6555, -1.6213,  ..., -2.0837, -2.0837, -1.9638]],
                
                [[ 0.2927,  0.3452,  0.2752,  ..., -0.3901, -0.4251, -0.7227],
                [ 0.3277,  0.3277,  0.2752,  ..., -0.4776, -0.5301, -0.5301],
                [ 0.4328,  0.3102,  0.2752,  ..., -0.3901, -0.6527, -0.6001],
                ...,
                [-1.1604, -1.0553, -0.7577,  ..., -0.8627, -1.0028, -1.1954],
                [-0.8102, -0.8277, -0.7052,  ..., -0.8627, -1.1954, -1.3880],
                [-0.3200, -0.2325, -0.2675,  ..., -0.9678, -1.3179, -1.1779]],
                
                [[-0.6193, -0.5147, -0.5495,  ..., -1.2816, -1.2990, -1.5081],
                [-0.5147, -0.5147, -0.5670,  ..., -1.2467, -1.2467, -1.2467],
                [-0.4275, -0.5321, -0.5495,  ..., -1.1944, -1.3339, -1.2816],
                ...,
                [-0.2010, -0.1487,  0.2522,  ...,  0.4962,  0.3045,  0.0953],
                [ 0.0256, -0.0267,  0.1651,  ...,  0.4788,  0.1476, -0.3578],
                [ 0.5136,  0.5311,  0.5136,  ...,  0.3219, -0.2010, -0.3055]]]],
                device='cuda:0')
                """
                # 2] Batch 단위 Target Data to cuda
                next_target = next_target.cuda(non_blocking=True) # next_target : shape(64)
                """
                tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
                """
                # 3] Batch 단위 Input Data 정규화
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            # 4] Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data)
            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream) # 역할 ???!!!
            input = next_input
            target = next_target

            # ==============================================================
            # {1} Bang Env - Load Time Check
            # ==============================================================
            load_time.update(time.time() - end)
            print((
                    f'load_count {idx + 1}/{len(loader)} | '
                    + f'load_time {load_time.avg}'
            ))
            if load_time.count % (len(loader)//10) == 0:
                print((
                        f'load_count {idx+1}/{len(loader)} | '
                        + f'load_time {load_time.avg}'
                ))
                load_time.reset()
            # ==============================================================
        yield input, target

    # 1]] DataLoader 초기화
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    # 2]] DataLoader 반복
    def __iter__(self):
        # 1. 분산 학습인 경우 (dataloader.sampler == torch.utils.data.distributed.DistributedSampler) : Epoch 설정
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        # 2. Epoch 증가
        self.epoch += 1
        # 3. Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data)
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

# --------------------------------------------------------------
# 10) DataLoader
    # (1) Data 불러오는 형식
        # 1] Data Backend == Pytorch인 경우
# --------------------------------------------------------------
def get_pytorch_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None, input_size=224):
    # [1] train dataset 경로 불러오기
    traindir = os.path.join(data_path, 'train')

    # [2] train dataset 이미지 불러오기 + Augmentation
    train_dataset = datasets.ImageFolder( # 구조!!!
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                ]))
    # print("train_dataset : ", train_dataset)
    """
    train_dataset :  Dataset ImageFolder
    Number of datapoints: 50000
    Root location: /home/hi/Datasets/Cifar10/train
    StandardTransform
    Transform: Compose(
                   RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bilinear), antialias=None)
                   RandomHorizontalFlip(p=0.5)
               )
    """

    # [3] Sampler 설정
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # 1]] 분산학습 하는 경우
    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # 2]] 분산학습 하지 않는 경우
    else:
        train_sampler = None

    # [4] train dataset 최종 불러오기 (batch size / shuffle / num_workers / worker_init_fn / pin_memory / sampler / collate_fn)
    train_loader = torch.utils.data.DataLoader( # 구조!!!
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    # ==============================================================
    # {1} Bang Env - Image Check
    # ==============================================================
    # 1) train_loader에서 호출 가능한 객체 반환
    train_iter = iter(train_loader)
    # 2) train_iter에서 input / output 정보 추출
    input, target = next(train_iter)
    # 3) 이미지 하나씩 설정
    for idx, i in enumerate(input):
        # (1) Plt Title
        plt.title(int(target[idx]))
        # (2) Plt Subplot Split
        plt.subplot(int(len(input)**0.5)+1, int(len(input)**0.5), idx + 1)
        # (3) Subplot에 Image 저장
        plt.imshow(tensor_to_pltimg(input[idx]))
        # (4) Plt Figure Axis
        plt.axis("off")
    # 4) 이미지 출력
    plt.show()
    # ==============================================================

    # print("train_iter : ", train_iter) # <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7f7a34623910>
    # print("input.size(), target.size() : ", input.size(), target.size()) # torch.Size([64, 3, 224, 224]) torch.Size([64])
    # print("target : ", target)
    """
    tensor([8, 5, 4, 9, 3, 5, 7, 8, 7, 9, 8, 1, 2, 3, 1, 6, 8, 7, 4, 0, 8, 1, 4, 9,
        2, 1, 2, 3, 8, 9, 7, 1, 6, 3, 3, 8, 7, 6, 7, 3, 1, 8, 7, 7, 2, 7, 3, 2,
        1, 3, 6, 0, 0, 3, 0, 8, 9, 9, 9, 1, 0, 9, 9, 1])
    """
    # ==============================================================

    # [5] Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data) + DataLoader 길이 불러오기
    return PrefetchedWrapper(train_loader), len(train_loader)

# --------------------------------------------------------------
# 10) DataLoader
    # (1) Data 불러오는 형식
        # 1] Data Backend == Pytorch인 경우
# --------------------------------------------------------------
def get_pytorch_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None, input_size=224):
    # [1] val dataset 경로 불러오기
    valdir = os.path.join(data_path, 'val')

    # [2] val dataset 이미지 불러오기 + Augmentation
    val_dataset = datasets.ImageFolder( # 구조!!!
            valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                ]))

    # [3] Sampler 설정
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # 1]] 분산학습 하는 경우
    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # 2]] 분산학습 하지 않는 경우
    else:
        val_sampler = None

    # [4] val dataset 최종 불러오기(batch size / shuffle / num_workers / worker_init_fn / pin_memory / sampler / collate_fn)
    val_loader = torch.utils.data.DataLoader( # 구조!!!
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=fast_collate)

    # [5] Dataloader에서 Batch 단위로 Data 불러오기 (Input Data, Target Data) + DataLoader 길이 불러오기
    return PrefetchedWrapper(val_loader), len(val_loader)