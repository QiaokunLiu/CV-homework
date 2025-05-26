import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import h5py
from tqdm import tqdm
import cv2
import torch

def gaussian_kernel(size, sigma):
    """生成高斯核"""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x*x + y*y)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
    return g / g.sum()

def generate_density_map(image_shape, points, kernel_size=15, sigma=4):
    """生成密度图"""
    density_map = np.zeros(image_shape[:2], dtype=np.float32)
    
    # 对每个标注点生成高斯核
    for point in points:
        x, y = int(point[0]), int(point[1])
        if x < image_shape[1] and y < image_shape[0]:
            density_map[y, x] = 1
    
    density_map = gaussian_filter(density_map, sigma=sigma)
    # 转换为PyTorch张量
    density_map = torch.from_numpy(density_map)
    return density_map

def process_dataset(root_path, phase='train'):
    """处理数据集"""
    # 创建输出目录
    image_dir = os.path.join(root_path, f'{phase}_data', 'images')
    gt_dir = os.path.join(root_path, f'{phase}_data', 'ground_truth')
    os.makedirs(gt_dir, exist_ok=True)
    
    # 读取ground truth文件
    gt_path = os.path.join(root_path, f'{phase}_data', 'ground_truth')
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    print(f'处理{phase}数据集...')
    for img_file in tqdm(image_files):
        # 读取图像尺寸
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        
        # 读取对应的ground truth
        mat_file = os.path.join(gt_path, "GT_"+img_file.replace('.jpg', '.mat'))
        if os.path.exists(mat_file):
            try:
                # 尝试不同的mat文件格式
                try:
                    mat = sio.loadmat(mat_file)
                    points = mat['image_info'][0, 0][0, 0][0]
                except:
                    mat = h5py.File(mat_file, 'r')
                    points = np.asarray(mat['image_info']['location']['x']).T
                
                # 生成密度图（现在返回的是PyTorch张量）
                density_map = generate_density_map(img.shape, points)
                
                # 保存为pt文件
                save_path = os.path.join(gt_dir, img_file.replace('.jpg', '.pt'))
                print(save_path)
                torch.save(density_map, save_path)
            except Exception as e:
                print(f'处理文件 {img_file} 时出错: {str(e)}')

def main():
    root_path = '.\\..\\data\\shanghaitech\\part_A'
    
    # 处理训练集和测试集
    process_dataset(root_path, 'train')
    process_dataset(root_path, 'test')
    
    print('数据预处理完成！')

if __name__ == '__main__':
    main() 