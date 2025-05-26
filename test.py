import torch
from torch.utils.data import DataLoader
from models.mobilenetv3 import CrowdCounter
from utils.dataset import ShanghaiTechDataset
from tqdm import tqdm
import numpy as np

def test():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载测试数据
    test_dataset = ShanghaiTechDataset('data/ShanghaiTech/part_A', phase='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 加载模型
    model = CrowdCounter().to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model.eval()
    
    # 测试指标
    mae = 0
    mse = 0
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            mae += torch.abs(outputs - targets).mean().item()
            mse += ((outputs - targets) ** 2).mean().item()
    
    mae /= len(test_loader)
    mse = np.sqrt(mse / len(test_loader))
    
    print(f'Test Results:')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {mse:.2f}')

if __name__ == '__main__':
    test() 