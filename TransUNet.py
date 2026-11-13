import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TransformerBottleneck(nn.Module):
    def __init__(self, channels, num_heads=8, depth=2, mlp_ratio=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * mlp_ratio,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14, channels))

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        pos = self.pos_embed
        if tokens.size(1) != pos.size(1):
            side = int(pos.size(1) ** 0.5)
            pos_2d = pos.transpose(1, 2).reshape(1, c, side, side)
            pos_2d = F.interpolate(pos_2d, size=(h, w), mode="bilinear", align_corners=False)
            pos = pos_2d.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + pos)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class TransUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)
        self.transformer = TransformerBottleneck(base_ch * 16, num_heads=8, depth=2)

        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out = nn.Sequential(nn.Conv2d(base_ch, out_ch, 1))
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.transformer(self.bottleneck(self.pool(e4)))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.out(d1)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return self.last_activation(logits)


if __name__ == "__main__":
    model = TransUNet(3, 1)
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)
