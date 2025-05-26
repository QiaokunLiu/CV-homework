import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ShanghaiTechDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 设置数据路径
        self.img_dir = os.path.join(root_dir, f'{phase}_data', 'images')
        self.gt_dir = os.path.join(root_dir, f'{phase}_data', 'ground_truth')
        
        # 获取所有图片文件名
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name.replace('.jpg', '.pt'))
        
        # 读取图片和ground truth
        image = Image.open(img_path).convert('RGB')
        density_map = torch.load(gt_path)
        
        # 应用图像变换
        image = self.transform(image)
        
        # 计算总人数
        count = torch.sum(density_map)
        
        return image, count 