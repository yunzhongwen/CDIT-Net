import os
import re
import sys
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from tqdm import tqdm
import utils.DataLoaderFM as DLr
from utils import Consistency
from utils.CUDA_Check import GPUorCPU
from torch.utils.data import DataLoader

# 导入你的网络模型
from network2 import PDRMultiFocusFusionNet


class Fusion:
    def __init__(self,
                 modelpath='D:/Image_fusionW/Image_fusion/RunTimeData/2026-10-30 01.58.38bt3/best_network.pth',   # 训练好的权重位置
                 dataroot=r'./Datasets/Eval',     # 测试数据集根目录
                 dataset_name='Lytro',             # 数据集名称
                 threshold=0.0015,                # 小区域移除阈值
                 ):
        raw_device = GPUorCPU().DEVICE
        self.DEVICE = torch.device(raw_device) if isinstance(raw_device, str) else raw_device

        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.SAVEPATH = os.path.join('./Results', self.DATASET_NAME)
        self.DATAPATH = os.path.join(self.DATAROOT, self.DATASET_NAME)

    def __call__(self, *args, **kwargs):
        MODEL = self.LoadWeights(self.MODELPATH)
        EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
        if len(EVAL_LIST_A) == 0:
            print("错误：未匹配到有效的测试图像对！请检查数据集路径。")
            return
        self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)

    # 加载已训练好的模型权重
    def LoadWeights(self, modelpath):
        model = PDRMultiFocusFusionNet().to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath, map_location=self.DEVICE), strict=False)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        print("The number of model parameters: {} M\n\n".format(round(num_params / 1e6, 6)))
        return model

    # 准备数据：兼容 -A/-B（如 Lytro）和 _A/_B（如 MFFW）来对齐图像
    def PrepareData(self, datapath):
        path_A = os.path.join(datapath, 'sourceA')
        path_B = os.path.join(datapath, 'sourceB')

        # 获取所有文件路径
        list_A = glob.glob(os.path.join(path_A, '*.*'))
        list_B = glob.glob(os.path.join(path_B, '*.*'))

        # 构建字典：去除文件名末尾的 -A, -a, _A, _a 作为键
        files_A = {}
        for f in list_A:
            stem = os.path.splitext(os.path.basename(f))[0]
            key = re.sub(r'[-_][Aa]$', '', stem)
            files_A[key] = f

        # 构建字典：去除文件名末尾的 -B, -b, _B, _b 作为键
        files_B = {}
        for f in list_B:
            stem = os.path.splitext(os.path.basename(f))[0]
            key = re.sub(r'[-_][Bb]$', '', stem)
            files_B[key] = f

        # 提取共同的核心文件名并排序
        common_keys = sorted(list(set(files_A.keys()) & set(files_B.keys())))

        # 根据对齐的键获取对应的路径
        eval_list_A = [files_A[k] for k in common_keys]
        eval_list_B = [files_B[k] for k in common_keys]

        print(f"成功匹配到 {len(eval_list_A)} 组对齐的测试图像。")
        return eval_list_A, eval_list_B

    # 一致性检验
    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = Consistency.Binarization(img_tensor)  # 二值化
        if threshold != 0:
            Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        os.makedirs(savepath, exist_ok=True)

        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)

        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
        running_time = []

        # 设定对齐倍数（通常网络有 4 次或 5 次下采样，设为 32 最稳妥）
        size_divisor = 32

        with torch.no_grad():
            for idx, (A, B) in enumerate(eval_loader_tqdm):
                start_time = time.time()

                A = A.to(self.DEVICE)
                B = B.to(self.DEVICE)

                # 1. 记录原始尺寸 (B, C, H, W)，并计算需要 Pad 的像素数
                _, _, H, W = A.shape
                pad_h = (size_divisor - H % size_divisor) % size_divisor
                pad_w = (size_divisor - W % size_divisor) % size_divisor

                if pad_h != 0 or pad_w != 0:
                    # F.pad 参数顺序为 (左, 右, 上, 下)，使用 reflect 镜像填充避免边界伪影
                    A = F.pad(A, (0, pad_w, 0, pad_h), mode='reflect')
                    B = F.pad(B, (0, pad_w, 0, pad_h), mode='reflect')

                # 2. 模型预测
                NetOut = model(A, B)
                if isinstance(NetOut, (tuple, list)):
                    map_final = NetOut[1]
                else:
                    map_final = NetOut

                # 3. 将预测出的特征图裁剪回原始尺寸 (H, W)
                map_final = map_final[:, :, :H, :W]

                # 一致性检验
                D = self.ConsisVerif(map_final, threshold)[0]

                # 正确将 PyTorch (C, H, W) 转换为 NumPy/OpenCV 支持的 (H, W, C)
                if D.dim() == 3:
                    D_np = D.permute(1, 2, 0).detach().cpu().numpy()
                else:
                    D_np = D.detach().cpu().numpy()
                    D_np = np.expand_dims(D_np, axis=-1)

                # 读取当前 Batch 对应的源图像
                img_A = cv2.imread(eval_list_A[idx])
                img_B = cv2.imread(eval_list_B[idx])

                # 尺寸不一致时安全调整
                if D_np.shape[:2] != img_A.shape[:2]:
                    D_np = cv2.resize(D_np, (img_A.shape[1], img_A.shape[0]))
                    if len(D_np.shape) == 2:
                        D_np = np.expand_dims(D_np, axis=-1)

                # 修复语法错误：执行加权融合
                Final_fused = img_A * D_np + img_B * (1 - D_np)

                # 提取原文件名并去除末尾的 -A / _A 后进行保存
                raw_name = os.path.splitext(os.path.basename(eval_list_A[idx]))[0]
                base_name = re.sub(r'[-_][Aa]$', '', raw_name)
                save_dm_path = os.path.join(savepath, f"{base_name}-dm.png")
                save_fused_path = os.path.join(savepath, f"{base_name}.png")

                cv2.imwrite(save_dm_path, (D_np * 255).astype(np.uint8))
                cv2.imwrite(save_fused_path, Final_fused.astype(np.uint8))

                running_time.append(time.time() - start_time)

        for i, r_time in enumerate(running_time):
            print("process_time: {:.5f} s".format(r_time))

        if len(running_time) > 0:
            avg_time = sum(running_time) / len(running_time)
            print("\nAverage process time per image: {:.5f} s".format(avg_time))

        print("\nResults are saved in: " + savepath)


if __name__ == '__main__':
    f = Fusion()
    f()