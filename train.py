import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.crowd_models import (
    CrowdCounter_Small,
    CrowdCounter_Small_Deep,
    CrowdCounter_Small_Attention,
    CrowdCounter_Small_CBAM
)
from utils.dataset import ShanghaiTechDataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_curves(histories, save_dir='checkpoints'):
    """绘制多个模型的训练曲线"""
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    for model_name, history in histories.items():
        plt.plot(history['train_losses'], label=f'{model_name} Train')
    plt.title('Train Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for model_name, history in histories.items():
        plt.plot(history['val_losses'], label=f'{model_name} Val')
    plt.title('Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制MAE曲线
    plt.subplot(2, 2, 3)
    for model_name, history in histories.items():
        plt.plot(history['train_maes'], label=f'{model_name} Train')
    plt.title('Train MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for model_name, history in histories.items():
        plt.plot(history['val_maes'], label=f'{model_name} Val')
    plt.title('Validation MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_curves.png'))
    plt.close()

def evaluate_model(model, data_loader, device):
    """评估模型"""
    model.eval()
    total_mae = 0
    total_mse = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            
            mae = torch.abs(outputs - targets).mean().item()
            mse = ((outputs - targets) ** 2).mean().item()
            
            total_mae += mae
            total_mse += mse
    
    # 合并所有预测和目标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    avg_mae = total_mae / len(data_loader)
    avg_rmse = np.sqrt(total_mse / len(data_loader))
    
    return avg_mae, avg_rmse, all_preds, all_targets

def train_model(model, model_name, train_loader, val_loader, device, num_epochs=100):
    """训练单个模型"""
    # 创建模型专属的保存目录
    model_save_dir = os.path.join('checkpoints', model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_maes': [],
        'val_maes': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_mae = 0
        
        for images, targets in tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.abs(outputs - targets).mean().item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_mae = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_mae += torch.abs(outputs - targets).mean().item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # 记录历史
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_maes'].append(train_mae)
        history['val_maes'].append(val_mae)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型和当前epoch的状态
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
            'history': history
        }
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(model_save_dir, 'best_model.pt'))
        
        # 每10个epoch保存一次checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(model_save_dir, f'epoch_{epoch+1}.pt'))
            # 保存当前的训练曲线
            plot_single_model_curves(history, model_name, model_save_dir)
        
        print(f'{model_name} Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.1e}\n')
    
    return history

def plot_single_model_curves(history, model_name, save_dir):
    """绘制单个模型的训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.title(f'{model_name} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制MAE曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_maes'], label='Train MAE')
    plt.plot(history['val_maes'], label='Val MAE')
    plt.title(f'{model_name} MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_dataset = ShanghaiTechDataset('data/shanghaitech/part_A', phase='train')
    val_dataset = ShanghaiTechDataset('data/shanghaitech/part_A', phase='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 创建主结果目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 定义要比较的模型
    models = {
        'MobileNetV3_Small': CrowdCounter_Small(),
        'MobileNetV3_Small_Deep': CrowdCounter_Small_Deep(),
        'MobileNetV3_Small_Attention': CrowdCounter_Small_Attention(),
        'MobileNetV3_Small_CBAM': CrowdCounter_Small_CBAM()
    }
    
    # 训练历史记录
    histories = {}
    final_results = {}
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n开始训练 {model_name}...")
        model = model.to(device)
        
        # 创建模型专属的保存目录
        model_save_dir = os.path.join('checkpoints', model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 训练模型
        history = train_model(model, model_name, train_loader, val_loader, device)
        histories[model_name] = history
        
        # 加载最佳模型进行测试
        checkpoint = torch.load(os.path.join(model_save_dir, 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        test_mae, test_rmse, preds, targets = evaluate_model(model, val_loader, device)
        
        final_results[model_name] = {
            'MAE': test_mae,
            'RMSE': test_rmse
        }
        
        # 保存预测结果
        torch.save({
            'predictions': preds,
            'targets': targets,
            'metrics': {
                'MAE': test_mae,
                'RMSE': test_rmse
            }
        }, os.path.join(model_save_dir, 'predictions.pt'))
        
        # 保存最终的训练曲线
        plot_single_model_curves(history, model_name, model_save_dir)
    
    # 绘制所有模型的对比曲线
    plot_curves(histories, save_dir='checkpoints')
    
    # 打印最终结果对比
    print("\n最终测试集结果对比:")
    print("-" * 50)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8}")
    print("-" * 50)
    for model_name, results in final_results.items():
        print(f"{model_name:<25} {results['MAE']:8.2f} {results['RMSE']:8.2f}")
    print("-" * 50)
    
    # 保存总体结果
    torch.save({
        'histories': histories,
        'final_results': final_results
    }, os.path.join('checkpoints', 'all_results.pt'))

if __name__ == '__main__':
    main() 