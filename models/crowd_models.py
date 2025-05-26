import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class CBAM(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        
        return x * spatial_att

class CrowdCounter_Small(nn.Module):
    """使用MobileNetV3-Small的人群计数模型"""
    def __init__(self):
        super(CrowdCounter_Small, self).__init__()
        base_model = mobilenet_v3_small(pretrained=True)
        self.features = base_model.features
        
        # MobileNetV3-Small的最后一层通道数是576
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

class CrowdCounter_Small_Deep(nn.Module):
    """使用更深的回归头的MobileNetV3-Small模型"""
    def __init__(self):
        super(CrowdCounter_Small_Deep, self).__init__()
        base_model = mobilenet_v3_small(pretrained=True)
        self.features = base_model.features
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

class CrowdCounter_Small_Attention(nn.Module):
    """使用简单通道注意力的MobileNetV3-Small模型"""
    def __init__(self):
        super(CrowdCounter_Small_Attention, self).__init__()
        base_model = mobilenet_v3_small(pretrained=True)
        self.features = base_model.features
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 144, 1),  # 576 // 4 = 144
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 576, 1),
            nn.Sigmoid()
        )
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        # 应用通道注意力
        attention = self.channel_attention(x)
        x = x * attention
        x = self.regressor(x)
        return x

class CrowdCounter_Small_CBAM(nn.Module):
    """使用CBAM注意力机制的MobileNetV3-Small模型"""
    def __init__(self):
        super(CrowdCounter_Small_CBAM, self).__init__()
        base_model = mobilenet_v3_small(pretrained=True)
        self.features = base_model.features
        
        # CBAM注意力模块
        self.cbam = CBAM(channels=576)
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 应用CBAM
        x = self.cbam(x)
        # 回归预测
        x = self.regressor(x)
        return x 