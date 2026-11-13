import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return windows.view(-1, window_size * window_size, c)


def window_reverse(windows, window_size, h, w, b):
    c = windows.shape[-1]
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, h, w)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class WindowAttentionBlock(nn.Module):
    def __init__(self, channels, window_size=7, num_heads=4, shift=False, mlp_ratio=4):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels),
        )

    def _shifted_window_mask(self, height, width, device):
        img_mask = torch.zeros((1, 1, height, width), device=device)
        shift = self.window_size // 2
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift),
            slice(-shift, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift),
            slice(-shift, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, :, h_slice, w_slice] = count
                count += 1

        mask_windows = window_partition(img_mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[2:]
        shift = self.window_size // 2 if self.shift else 0
        if shift:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))

        windows = window_partition(x, self.window_size)
        attn_in = self.norm1(windows)
        attn_mask = None
        if shift:
            attn_mask = self._shifted_window_mask(hp, wp, x.device)
            attn_mask = attn_mask.repeat(b, self.attn.num_heads, 1, 1)
            attn_mask = attn_mask.view(-1, self.window_size * self.window_size, self.window_size * self.window_size)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=attn_mask, need_weights=False)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))
        x = window_reverse(windows, self.window_size, hp, wp, b)

        if shift:
            x = torch.roll(x, shifts=(shift, shift), dims=(2, 3))
        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        return x


class SwinStage(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.blocks = nn.Sequential(
            WindowAttentionBlock(channels, num_heads=num_heads, shift=False),
            WindowAttentionBlock(channels, num_heads=num_heads, shift=True),
        )

    def forward(self, x):
        return self.blocks(x)


class SwinUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = nn.Sequential(ConvBlock(in_ch, base_ch), SwinStage(base_ch, 4))
        self.enc2 = nn.Sequential(ConvBlock(base_ch, base_ch * 2), SwinStage(base_ch * 2, 4))
        self.enc3 = nn.Sequential(ConvBlock(base_ch * 2, base_ch * 4), SwinStage(base_ch * 4, 8))
        self.enc4 = nn.Sequential(ConvBlock(base_ch * 4, base_ch * 8), SwinStage(base_ch * 8, 8))
        self.bottleneck = nn.Sequential(ConvBlock(base_ch * 8, base_ch * 16), SwinStage(base_ch * 16, 8))

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
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.out(d1)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return self.last_activation(logits)


if __name__ == "__main__":
    model = SwinUNet(3, 1)
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)
