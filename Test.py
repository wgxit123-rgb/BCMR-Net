# Test_Segmentation.py
"""
语义分割二分类测试脚本
支持多种评价指标：IoU, Dice, Accuracy, Precision, Recall, Specificity, F1-Score
"""

import torch
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, jaccard_score
)
from scipy.spatial.distance import directed_hausdorff
from boundary_metrics import robust_boundary_metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")
from Load_Dataset import ImageToImage2D, ValGenerator
# 导入你的模型和数据加载器
from UNet import UNet
#from ResU_Net import ResUnet
#from DeepLabv3plus import DeepLabV3Plus
#from  UNetPlusPlus import NestedUNet
from SE_UNet import SE_UNet
from SE_UNet_MK_KAN_RPA import SE_UNet_MK_KAN_RPA
from ResU_Net import ResUnet
from DeepLabv3plus import DeepLabV3Plus
from UNetPlusPlus import NestedUNet
from TransUNet import TransUNet
from SwinUNet import SwinUNet
from SE_UNet_Ablation import (
    SE_UNet_MK,
    SE_UNet_KAN,
    SE_UNet_RPA,
    SE_UNet_MK_KAN,
    SE_UNet_MK_RPA,
    SE_UNet_KAN_RPA,
)
from model_factory import create_model, FACTORY_MODEL_CHOICES


MODEL_CHOICES = [
    "UNet",
    "UNetPlusPlus",
    "UNet++",
    "DeepLabV3Plus",
    "DeepLabv3+",
    "ResUnet",
    "ResUNet",
    "SE_UNet",
    "SEUNet",
    "SwinUNet",
    "Swin-Unet",
    "TransUNet",
    "TransUnet",
    "SE_UNet_MK",
    "SE_UNet_KAN",
    "SE_UNet_RPA",
    "SE_UNet_MK_KAN",
    "SE_UNet_MK_RPA",
    "SE_UNet_KAN_RPA",
    "SE_UNet_MK_KAN_RPA",
    "SEUNet_MK_KAN_RPA",
    "SE_MK_KAN_RPA",
]
MODEL_CHOICES = sorted(set(MODEL_CHOICES + FACTORY_MODEL_CHOICES))




class SegmentationMetrics:
    """语义分割评价指标计算类"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.total_dice = 0
        self.total_iou = 0
        self.total_accuracy = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_specificity = 0
        self.total_f1 = 0
        self.total_hausdorff = 0
        self.total_hd95 = 0
        self.total_asd = 0
        self.total_boundary_f1 = 0
        self.num_samples = 0

        # 用于存储每个样本的指标
        self.per_sample_metrics = []

    def calculate_metrics(self, pred, target, threshold=0.5):
        """
        计算单个样本的所有指标

        Args:
            pred: 预测结果 (H, W) 或 (C, H, W)
            target: 真实标签 (H, W) 或 (C, H, W)
            threshold: 二值化阈值

        Returns:
            dict: 包含所有指标的字典
        """
        # 转换为numpy数组
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        # 处理维度
        if len(pred.shape) > 2:
            pred = pred.squeeze()
        if len(target.shape) > 2:
            target = target.squeeze()

        # 二值化预测结果
        pred_binary = (pred > threshold).astype(np.uint8)
        target_binary = (target > 0.5).astype(np.uint8)

        # 展平
        pred_flat = pred_binary.flatten()
        target_flat = target_binary.flatten()

        # 计算混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0, 1]).ravel()

        # 1. Dice系数
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        # 2. IoU (Jaccard)
        iou = tp / (tp + fp + fn + 1e-8)

        # 3. 准确率
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        # 4. 精确率
        precision = tp / (tp + fp + 1e-8)

        # 5. 召回率（灵敏度）
        recall = tp / (tp + fn + 1e-8)

        # 6. 特异性
        specificity = tn / (tn + fp + 1e-8)

        # 7. F1分数
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # 8. Hausdorff距离（可选，计算较慢）
        try:
            if np.sum(pred_binary) > 0 and np.sum(target_binary) > 0:
                # 提取边界点
                pred_contour = cv2.Canny(pred_binary * 255, 100, 200) > 0
                target_contour = cv2.Canny(target_binary * 255, 100, 200) > 0

                if np.sum(pred_contour) > 0 and np.sum(target_contour) > 0:
                    pred_contour_pts = np.argwhere(pred_contour)
                    target_contour_pts = np.argwhere(target_contour)

                    hausdorff = max(
                        directed_hausdorff(pred_contour_pts, target_contour_pts)[0],
                        directed_hausdorff(target_contour_pts, pred_contour_pts)[0]
                    )
                else:
                    hausdorff = 0
            else:
                hausdorff = 0
        except:
            hausdorff = 0

        boundary_metrics = robust_boundary_metrics(
            pred_binary,
            target_binary,
            tolerance_px=2.0,
        )

        metrics = {
            'Dice': dice,
            'IoU': iou,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1': f1,
            'Hausdorff': hausdorff,
            'HD95': boundary_metrics['HD95'],
            'ASD': boundary_metrics['ASD'],
            'BoundaryF1': boundary_metrics['BoundaryF1'],
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        }

        return metrics

    def update(self, pred, target, threshold=0.5, sample_name=None):
        """
        更新累积指标

        Args:
            pred: 预测结果
            target: 真实标签
            threshold: 二值化阈值
        """
        metrics = self.calculate_metrics(pred, target, threshold)

        self.total_dice += metrics['Dice']
        self.total_iou += metrics['IoU']
        self.total_accuracy += metrics['Accuracy']
        self.total_precision += metrics['Precision']
        self.total_recall += metrics['Recall']
        self.total_specificity += metrics['Specificity']
        self.total_f1 += metrics['F1']
        self.total_hausdorff += metrics['Hausdorff']
        self.total_hd95 += metrics['HD95']
        self.total_asd += metrics['ASD']
        self.total_boundary_f1 += metrics['BoundaryF1']
        self.num_samples += 1

        if sample_name is not None:
            metrics = {'Sample': str(sample_name), **metrics}

        self.per_sample_metrics.append(metrics)

    def get_average_metrics(self):
        """获取平均指标"""
        if self.num_samples == 0:
            return {}

        return {
            'Dice': self.total_dice / self.num_samples,
            'IoU': self.total_iou / self.num_samples,
            'Accuracy': self.total_accuracy / self.num_samples,
            'Precision': self.total_precision / self.num_samples,
            'Recall': self.total_recall / self.num_samples,
            'Specificity': self.total_specificity / self.num_samples,
            'F1': self.total_f1 / self.num_samples,
            'Hausdorff': self.total_hausdorff / self.num_samples,
            'HD95': self.total_hd95 / self.num_samples,
            'ASD': self.total_asd / self.num_samples,
            'BoundaryF1': self.total_boundary_f1 / self.num_samples,
        }

    def print_metrics(self, metrics_dict):
        """格式化打印指标"""
        print("\n" + "=" * 60)
        print(" " * 15 + "Evaluation Metrics")
        print("=" * 60)
        for metric_name, value in metrics_dict.items():
            if metric_name in ['TP', 'TN', 'FP', 'FN']:
                continue
            print(f"  {metric_name:<15}: {value:.4f}")
        print("=" * 60)


def load_model(model_path, in_ch=3, out_ch=1, device='cuda', model_name='SE_UNet'):
    """
    加载训练好的模型

    Args:
        model_path: 模型权重路径
        in_ch: 输入通道数
        out_ch: 输出通道数
        device: 设备
    """
    if model_name in FACTORY_MODEL_CHOICES:
        model = create_model(model_name, in_ch=in_ch, out_ch=out_ch)
    elif model_name == 'UNet':
        model = UNet(in_ch, out_ch)
    elif model_name in ('UNetPlusPlus', 'UNet++'):
        model = NestedUNet(in_channels=in_ch, num_classes=out_ch,
                           deep_supervision=True, bilinear=True)
    elif model_name in ('DeepLabV3Plus', 'DeepLabv3+'):
        model = DeepLabV3Plus(
            n_classes=out_ch,
            n_blocks=[3, 4, 6, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,
        )
    elif model_name in ('ResUnet', 'ResUNet'):
        model = ResUnet(in_channels=in_ch, num_classes=out_ch, base_c=32, bilinear=True)
    elif model_name in ('SE_UNet', 'SEUNet'):
        model = SE_UNet(in_ch=in_ch, out_ch=out_ch)
    elif model_name in ('SwinUNet', 'Swin-Unet'):
        model = SwinUNet(in_ch=in_ch, out_ch=out_ch, base_ch=32)
    elif model_name in ('TransUNet', 'TransUnet'):
        model = TransUNet(in_ch=in_ch, out_ch=out_ch, base_ch=32)
    elif model_name == 'SE_UNet_MK':
        model = SE_UNet_MK(in_ch=in_ch, out_ch=out_ch)
    elif model_name == 'SE_UNet_KAN':
        model = SE_UNet_KAN(in_ch=in_ch, out_ch=out_ch)
    elif model_name == 'SE_UNet_RPA':
        model = SE_UNet_RPA(in_ch=in_ch, out_ch=out_ch)
    elif model_name == 'SE_UNet_MK_KAN':
        model = SE_UNet_MK_KAN(in_ch=in_ch, out_ch=out_ch)
    elif model_name == 'SE_UNet_MK_RPA':
        model = SE_UNet_MK_RPA(in_ch=in_ch, out_ch=out_ch)
    elif model_name == 'SE_UNet_KAN_RPA':
        model = SE_UNet_KAN_RPA(in_ch=in_ch, out_ch=out_ch)
    elif model_name in ('SE_UNet_MK_KAN_RPA', 'SEUNet_MK_KAN_RPA', 'SE_MK_KAN_RPA'):
        model = SE_UNet_MK_KAN_RPA(in_ch=in_ch, out_ch=out_ch)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    #model = ResUnet(in_channels=3, num_classes=1, base_c=32, bilinear=True)
    #model = NestedUNet(in_channels=3, num_classes=1, deep_supervision=True, bilinear=True)
    #model = E_SegNet_2D(model_name="mobilenetv4_conv_medium.e500_r224_in1k", image_size=224)

    if model_path is not None and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)

        # 处理不同的保存格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 处理DataParallel保存的模型
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 移除'module.'前缀
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print(f"Model loaded from: {model_path}")
    else:
        print("No pretrained model found, using random initialization")

    model = model.to(device)
    model.eval()
    return model


def create_test_dataloader(test_dataset_path, batch_size=1, image_size=224):
    """
    创建测试数据加载器

    Args:
        test_dataset_path: 测试数据集路径
        batch_size: 批大小
        image_size: 图像大小
    """
    val_tf = ValGenerator(output_size=[image_size, image_size])
    test_dataset = ImageToImage2D(test_dataset_path, val_tf, image_size=image_size)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return test_loader


def save_prediction_visualization(image, mask, pred, save_path, filename, threshold=0.5):
    """
    保存预测结果可视化

    Args:
        image: 原始图像 (C, H, W)
        mask: 真实标签 (H, W)
        pred: 预测结果 (H, W)
        save_path: 保存路径
        filename: 文件名
        threshold: 二值化阈值
    """
    os.makedirs(save_path, exist_ok=True)

    # 转换tensor到numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()

    # 处理维度
    if len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    mask = mask.squeeze()
    pred = pred.squeeze()

    # 二值化预测
    pred_binary = (pred > threshold).astype(np.float32)

    # 创建对比图像
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 原图
    if image.shape[-1] == 1:
        axes[0].imshow(image.squeeze(), cmap='gray')
    else:
        # 反归一化（如果需要）
        if image.max() <= 1.0:
            image = image * 255
        image = image.astype(np.uint8)
        axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 真实标签
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # 预测结果（概率图）
    axes[2].imshow(pred, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('Prediction (Probability)')
    axes[2].axis('off')

    # 预测结果（二值化）
    axes[3].imshow(pred_binary, cmap='gray')
    axes[3].set_title('Prediction (Binary)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename.replace('.png', '_result.png')),
                dpi=150, bbox_inches='tight')
    plt.close()


def val_model(model, test_loader, device='cuda', threshold=0.5,
               save_vis=False, vis_save_path='./test_results/E_SegNet_2D/'):
    """
    测试模型并计算所有指标

    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        threshold: 二值化阈值
        save_vis: 是否保存可视化结果
        vis_save_path: 可视化保存路径

    Returns:
        tuple: (平均指标字典, 每个样本的指标列表)
    """
    metrics_calculator = SegmentationMetrics()

    print(f"\nTesting on {len(test_loader.dataset)} images...")
    print(f"Threshold: {threshold}")

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")

        for batch_idx, (sampled_batch, names) in enumerate(progress_bar):
            # 获取数据
            images = sampled_batch['image'].to(device)
            masks = sampled_batch['label'].to(device)

            # 预测
            outputs = model(images)
            if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
                outputs = torch.sigmoid(outputs.float())

            # 对batch中的每个样本计算指标
            for i in range(images.size(0)):
                pred = outputs[i].squeeze()
                mask = masks[i].squeeze()

                # 更新指标
                metrics_calculator.update(
                    pred,
                    mask,
                    threshold,
                    sample_name=names[i],
                )

                # 保存可视化
                if save_vis and batch_idx < 10:  # 只保存前10个样本
                    save_prediction_visualization(
                        images[i], masks[i], outputs[i],
                        vis_save_path, names[i], threshold
                    )

            # 更新进度条
            avg_metrics = metrics_calculator.get_average_metrics()
            progress_bar.set_postfix({
                'Dice': f"{avg_metrics.get('Dice', 0):.4f}",
                'IoU': f"{avg_metrics.get('IoU', 0):.4f}"
            })

    # 计算最终平均指标
    final_metrics = metrics_calculator.get_average_metrics()

    return final_metrics, metrics_calculator.per_sample_metrics


def save_results_to_csv(average_metrics, per_sample_metrics, save_path='./test_results/'):
    """
    保存测试结果到CSV文件

    Args:
        average_metrics: 平均指标字典
        per_sample_metrics: 每个样本的指标列表
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 保存平均指标
    avg_df = pd.DataFrame([average_metrics])
    avg_df.to_csv(os.path.join(save_path, 'average_metrics.csv'), index=False)

    # 保存每个样本的指标
    sample_df = pd.DataFrame(per_sample_metrics)
    sample_df.to_csv(os.path.join(save_path, 'per_sample_metrics.csv'), index=False)

    print(f"\nResults saved to: {save_path}")


def plot_metrics_distribution(per_sample_metrics, save_path='./test_results/E_SegNet_2D/'):
    """
    绘制指标分布图

    Args:
        per_sample_metrics: 每个样本的指标列表
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame(per_sample_metrics)

    # 选择要绘制的指标
    metrics_to_plot = ['Dice', 'IoU', 'Accuracy', 'Precision', 'Recall', 'F1']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            ax = axes[idx]
            df[metric].hist(bins=20, ax=ax, edgecolor='black')
            ax.set_title(f'{metric} Distribution')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.axvline(df[metric].mean(), color='red', linestyle='--',
                       label=f'Mean: {df[metric].mean():.3f}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_distribution.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SE_UNet", choices=MODEL_CHOICES)
    parser.add_argument("--model-path", default="model/SEUnet.tar")
    parser.add_argument("--test-dataset", default="./datasets/split_dataset/test")
    parser.add_argument("--result-save-path", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-vis", action="store_true", default=False)
    args = parser.parse_args()
    """主测试函数"""
    # ============ 配置参数 ============
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 模型参数（需要与训练时保持一致）
    MODEL_PATH = args.model_path
    IN_CHANNELS = 3  # 输入通道数
    OUT_CHANNELS = 1  # 输出通道数（二分类为1）
    IMAGE_SIZE = args.image_size
    THRESHOLD = args.threshold

    # 测试数据集路径
    TEST_DATASET_PATH = args.test_dataset

    # 是否保存可视化结果
    SAVE_VISUALIZATION = args.save_vis

    # 结果保存路径
    RESULT_SAVE_PATH = args.result_save_path or f'./test_results/{args.model}/'

    # ============ 加载模型 ============
    print("\n" + "=" * 60)
    print("Loading model...")
    model = load_model(MODEL_PATH, IN_CHANNELS, OUT_CHANNELS, device, args.model)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============ 创建数据加载器 ============
    print("\n" + "=" * 60)
    print("Loading test dataset...")
    test_loader = create_test_dataloader(
        TEST_DATASET_PATH,
        batch_size=args.batch_size,
        image_size=IMAGE_SIZE,
    )
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # ============ 运行测试 ============
    print("\n" + "=" * 60)
    print("Starting evaluation...")

    # 测试模型
    average_metrics, per_sample_metrics = val_model(
        model,
        test_loader,
        device,
        threshold=THRESHOLD,
        save_vis=SAVE_VISUALIZATION,
        vis_save_path=os.path.join(RESULT_SAVE_PATH, 'visualizations/')
    )

    # ============ 打印结果 ============
    metrics_calculator = SegmentationMetrics()
    metrics_calculator.print_metrics(average_metrics)

    # 打印额外的统计信息
    print(f"\nTotal test samples: {len(per_sample_metrics)}")

    # 统计Dice分布
    dice_scores = [m['Dice'] for m in per_sample_metrics]
    print(f"\nDice Score Statistics:")
    print(f"  Mean: {np.mean(dice_scores):.4f}")
    print(f"  Std:  {np.std(dice_scores):.4f}")
    print(f"  Min:  {np.min(dice_scores):.4f}")
    print(f"  Max:  {np.max(dice_scores):.4f}")
    print(f"  Median: {np.median(dice_scores):.4f}")

    # ============ 保存结果 ============
    # 保存到CSV
    save_results_to_csv(average_metrics, per_sample_metrics, RESULT_SAVE_PATH)

    # 绘制指标分布图
    plot_metrics_distribution(per_sample_metrics, RESULT_SAVE_PATH)

    # 保存详细报告
    report_path = os.path.join(RESULT_SAVE_PATH, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Semantic Segmentation Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test Dataset: {TEST_DATASET_PATH}\n")
        f.write(f"Number of test samples: {len(per_sample_metrics)}\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Average Metrics:\n")
        f.write("-" * 60 + "\n")
        for metric, value in average_metrics.items():
            if metric not in ['TP', 'TN', 'FP', 'FN']:
                f.write(f"  {metric:<15}: {value:.6f}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("Dice Score Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Mean:   {np.mean(dice_scores):.6f}\n")
        f.write(f"  Std:    {np.std(dice_scores):.6f}\n")
        f.write(f"  Min:    {np.min(dice_scores):.6f}\n")
        f.write(f"  Max:    {np.max(dice_scores):.6f}\n")
        f.write(f"  Median: {np.median(dice_scores):.6f}\n")

    print(f"\nDetailed report saved to: {report_path}")
    print("\nTesting completed!")


if __name__ == "__main__":
    main()
