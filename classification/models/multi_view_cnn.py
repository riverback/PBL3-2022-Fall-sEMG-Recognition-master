import torch
from torch import nn


class Multiview_projection(nn.Module):
    def __init__(self, views):
        super().__init__()
        self.views = views
        self.project = nn.Conv2d(in_channels=1, out_channels=self.views, kernel_size=1, padding=0, stride=1)
        self.bn1 = nn.BatchNorm2d(self.views)
        self.conv = nn.Conv2d(in_channels=self.views, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        x = self.project(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class Multi_view_backbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.local_fc = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
    def forward(self, x):
        B = x.size(0)
        x = self.conv(x)
        x = self.local_fc(x)
        x = self.gap(x)
        x = self.fc(x.reshape(B, -1))
        return x

class Early_fusion(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.local_fc = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(0.2),
        )
        self.out = nn.Linear(64, classes)
        
    def forward(self, x):
        B = x.size(0)
        x = self.conv(x)
        x = self.local_fc(x)
        x = self.gap(x)
        x = self.mlp(x.reshape(B, -1))
        out = self.out(x)
        
        return out

class Late_fusion(nn.Module):
    def __init__(self, classes):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Dropout(0.2)
        )
        self.out = nn.Linear(64, classes)
    
    def forward(self, x):
        B = x.size(0)
        x = self.net(x.reshape(B, -1))
        out = self.out(x)
        return out
        

class Multi_View_CNN(nn.Module):
    
    def __init__(self, views, classes):
        super().__init__()
        
        self.project = Multiview_projection(views)
        self.early_fusion = Early_fusion(in_channels=64, classes=classes)
        self.late_fusion = Late_fusion(classes)
        self.multi_backbone = Multi_view_backbone(64)
        
    def forward(self, x):
        views = self.project(x)
        early_out = self.early_fusion(views)
        features = self.multi_backbone(views)
        late_out = self.late_fusion(features)
        decision_fusion = torch.add(early_out, late_out)
        
        return decision_fusion
        
    
    

