from __future__ import print_function, absolute_import
import torch

__all__ = ['accuracy']

# --------------------------------------------------------------
# 12) Train & Update Logger & Update Checkpoint
    # (2) Epoch 순회 (Update Train / Update Evaluate / LR / Save Checkpoint)
        # 1] Train -> loss, accuracy 출력
            # [3] Train 계산 + 업데이트 (Learning Rate / 소요 시간 / Output / Loss / Accuracy / Gradient Update)
                # 5]] Accuracy + Loss 계산
# --------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # print("output : ", output) # output : shape(64, 1000)
    """
    tensor([[-1.0690e-10, -1.7880e-11, -6.2340e-11,  ...,  8.8764e-10,
             -2.1612e-10,  2.3619e-10],
            [-3.2675e-11, -8.4471e-11, -1.2320e-10,  ...,  1.3518e-09,
             -4.2591e-10,  4.9529e-10],
            [-4.2651e-11, -2.2243e-10, -5.6650e-11,  ...,  1.8159e-09,
             -4.8853e-10,  6.5685e-10],
            ...,
            [ 3.8335e-11, -5.7388e-11, -5.1619e-11,  ...,  1.5860e-09,
             -5.1943e-10,  6.5020e-10],
            [-1.1572e-10, -1.4074e-10, -1.6167e-10,  ...,  1.2743e-09,
             -3.8800e-10,  4.2598e-10],
            [-8.0254e-11, -1.6113e-10, -2.5040e-11,  ...,  9.5463e-10,
             -3.2903e-10,  4.4762e-10]], device='cuda:0')
    """
    # print("target : ", target) # target : shape(64, 1)
    """
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
    """
    with torch.no_grad():
        # 1. 초기 변수 설정 (Accuracy 계산할 개수 / BatchSize)
        maxk = max(topk) # maxk : 5 / topk : (1, 5)
        batch_size = target.size(0) # batch_size : 64

        # 2. model의 classifier 결과 중에서, Top 5개 예측값 추출
        _, pred = output.topk(maxk, 1, True, True) # pred : shape(64, 5)
        pred = pred.t() # pred : shape(5, 64)

        # 3. Top 5개 정답률 계산 (예측값 <-> 실제값 비교)
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # correct : shape(5, 64)
        res = [] # res : shape(tensor(0), tensor(0))

        # 4.
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0) # 기존 : correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size)) # correct_k.mul_ : shape(1)
        return res