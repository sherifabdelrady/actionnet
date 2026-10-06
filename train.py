"""ActionNet — Video Action Recognition (SlowFast-inspired)"""
import torch, torch.nn as nn, argparse

class SlowFastBlock(nn.Module):
    def __init__(self, slow_ch, fast_ch):
        super().__init__()
        self.slow = nn.Sequential(nn.Conv3d(slow_ch, slow_ch, (1,3,3), padding=(0,1,1)), nn.BatchNorm3d(slow_ch), nn.ReLU())
        self.fast = nn.Sequential(nn.Conv3d(fast_ch, fast_ch, (3,3,3), padding=1), nn.BatchNorm3d(fast_ch), nn.ReLU())
        self.lateral = nn.Conv3d(fast_ch, slow_ch, (5,1,1), stride=(8,1,1), padding=(2,0,0))
    def forward(self, slow, fast):
        return self.slow(slow) + self.lateral(fast), self.fast(fast)

class ActionNet(nn.Module):
    NUM_CLASSES = 400
    def __init__(self):
        super().__init__()
        self.block1 = SlowFastBlock(64, 8)
        self.block2 = SlowFastBlock(64, 8)
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.head   = nn.Linear(64 + 8, self.NUM_CLASSES)
    def forward(self, slow, fast):
        s, f = self.block1(slow, fast); s, f = self.block2(s, f)
        s = self.pool(s).flatten(1); f = self.pool(f).flatten(1)
        return self.head(torch.cat([s, f], dim=1))

if __name__ == "__main__":
    model = ActionNet()
    slow = torch.zeros(1, 64, 8, 224, 224); fast = torch.zeros(1, 8, 64, 224, 224)
    out = model(slow, fast); print(f"ActionNet output: {out.shape} (batch=1, classes=400)")
