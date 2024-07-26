import time
import os.path
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_unet import UnetGenerator
from datasets import CityscapesDataset

# 定义类别映射
CLASS_MAPPING = {
    0: 0,  # background
    1: 1,  # road
    2: 2,  # sky
    3: 3,  # car
}


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    inference_times = []

    class_counts_gt = np.zeros(num_classes)
    class_counts_pred = np.zeros(num_classes)

    with torch.no_grad():
        for i, (imagergb, label_class, _) in enumerate(dataloader):
            x = imagergb.to(device)
            y_true = label_class.to(device)

            if i == 0:
                print(f"Image shape: {x.shape}")
                print(f"Label shape: {y_true.shape}")
                print(f"Unique labels in first batch: {torch.unique(y_true)}")

            start_time = time.time()
            y_pred = model(x)
            end_time = time.time()

            inference_times.append(end_time - start_time)

            y_pred = torch.argmax(y_pred, dim=1)

            if i == 0:
                print(f"Prediction shape: {y_pred.shape}")
                print(f"Unique predictions in first batch: {torch.unique(y_pred)}")

            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()

            # 只考虑4个类别
            mask = np.isin(y_true, list(CLASS_MAPPING.keys()))
            y_true_masked = y_true[mask]
            y_pred_masked = y_pred[mask]

            hist += fast_hist(y_true_masked, y_pred_masked, num_classes)

            class_counts_gt += np.bincount(y_true_masked, minlength=num_classes)
            class_counts_pred += np.bincount(y_pred_masked, minlength=num_classes)

    ious = per_class_iu(hist)
    mean_iou = np.mean(ious)
    avg_inference_time = sum(inference_times) / len(inference_times)

    return mean_iou, ious, class_counts_gt, class_counts_pred, avg_inference_time


def map_keys(key):
    # 创建一个映射字典来处理层名称的差异
    mapping = {
        r'down_(\d+)\.0\.': r'down_\1.0.0.',
        r'down_(\d+)\.1\.': r'down_\1.0.1.',
        r'down_(\d+)\.2\.': r'down_\1.1.',
        r'down_(\d+)\.3\.': r'down_\1.2.',
        r'bridge\.0\.': r'bridge.0.0.',
        r'bridge\.1\.': r'bridge.0.1.',
        r'bridge\.2\.': r'bridge.1.',
        r'bridge\.3\.': r'bridge.2.',
        r'up_(\d+)\.0\.': r'up_\1.0.0.',
        r'up_(\d+)\.1\.': r'up_\1.0.1.',
        r'up_(\d+)\.2\.': r'up_\1.1.',
        r'up_(\d+)\.3\.': r'up_\1.2.',
    }

    for pattern, replacement in mapping.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    return key


def is_quantized_tensor(tensor):
    return hasattr(tensor, 'qscheme')


def adjust_param_shape(param, target_shape):
    if param.shape == target_shape:
        return param
    elif len(param.shape) == 1 and len(target_shape) == 4:
        # 将1D参数扩展为4D
        return param.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(target_shape)
    elif len(param.shape) == 4 and len(target_shape) == 1:
        # 将4D参数压缩为1D
        return param.mean(dim=(1, 2, 3))
    else:
        print(f"Warning: Unable to adjust parameter shape from {param.shape} to {target_shape}")
        return param


def load_state_dict(model, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = re.sub(r'^module\.', '', k)
        new_k = map_keys(new_k)
        if is_quantized_tensor(v):
            v = v.dequantize()

        if new_k in model.state_dict():
            target_shape = model.state_dict()[new_k].shape
            v = adjust_param_shape(v, target_shape)

        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str,
                        default="C:\\Users\\26375\\Downloads\\DeepLabV3Plus-Pytorch-master\\datasets\\data\\cityscapes",
                        help="Cityscapes dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    img_data = CityscapesDataset(args.datadir, split='val', mode='fine', augment=False)
    img_batch = DataLoader(img_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Loaded {len(img_data)} validation images")
    print(f"Number of classes: {len(CLASS_MAPPING)}")

    # 加载32位浮点模型
    fp32_model = UnetGenerator(3, len(CLASS_MAPPING), 64).to(device)
    state_dict = torch.load("C:\\Users\\26375\\Downloads\\pytorch_unet-master\\unet.pt", map_location=device)
    fp32_model = load_state_dict(fp32_model, state_dict)

    # 加载Int8量化模型
    int8_model = UnetGenerator(3, len(CLASS_MAPPING), 64).to(device)
    state_dict = torch.load("C:\\Users\\26375\\Downloads\\pytorch_unet-master\\model_int8.pth", map_location=device)
    int8_model = load_state_dict(int8_model, state_dict)

    # 评估32位浮点模型
    print("\nEvaluating 32-bit float model...")
    fp32_miou, fp32_ious, fp32_gt_counts, fp32_pred_counts, fp32_time = evaluate_model(fp32_model, img_batch, device, len(CLASS_MAPPING))

    # 评估Int8量化模型
    print("\nEvaluating Int8 quantized model...")
    int8_miou, int8_ious, int8_gt_counts, int8_pred_counts, int8_time = evaluate_model(int8_model, img_batch, device, len(CLASS_MAPPING))

    # 输出结果
    def print_results(model_name, miou, ious, gt_counts, pred_counts, avg_time):
        print(f"\n{model_name}:")
        print(f"    - Mean IoU: {miou:.4f}")
        #print("    - Per-class IoU:")
        class_names = ['background', 'road', 'sky', 'car']
        #for i, iou in enumerate(ious):
            #print(f"        {class_names[i]}: {iou:.4f}")
        #print("    - Ground Truth class counts:")
        #for i, count in enumerate(gt_counts):
            #print(f"        {class_names[i]}: {count}")
        #print("    - Prediction class counts:")
        #for i, count in enumerate(pred_counts):
            #print(f"        {class_names[i]}: {count}")
        print(f"    - Average Inference Time: {avg_time:.4f} seconds")

    print_results("32-bit Float Model", fp32_miou, fp32_ious, fp32_gt_counts, fp32_pred_counts, fp32_time)
    print_results("Int8 Quantized Model", int8_miou, int8_ious, int8_gt_counts, int8_pred_counts, int8_time)

if __name__ == '__main__':
    main()