"""Models: a from-scratch 2-D Fourier Neural Operator and two conventional
convolutional baselines (U-Net, plain CNN) parameter-matched to it.

All models map (B, 3, N, N) -> (B, 1, N, N): input channels are the field
plus two coordinate channels (appended by the data pipeline), and they are
fully convolutional / spectral so they run at any resolution. Only the FNO,
however, is discretisation-invariant: its spectral weights live on a fixed
set of Fourier modes of the *domain*, while a CNN kernel is fixed in pixels
and therefore represents a different physical stencil at each resolution.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Pointwise linear map on the lowest Fourier modes (Li et al., 2021)."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch, self.weight1.shape[1], h, w // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weight1
        )
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weight2
        )
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2d(nn.Module):
    """4-layer FNO2d with a 1x1-conv bypass path per spectral block.

    pad_frac > 0 zero-pads the lifted representation by that fraction of the
    grid before the spectral blocks (and crops after), which lets the
    Fourier layers handle non-periodic (Dirichlet) problems. The pad is
    proportional to the resolution, so it represents a fixed fraction of the
    physical domain at any grid size.
    """

    def __init__(self, in_channels=3, out_channels=1, modes=12, width=32,
                 n_layers=4, pad_frac=0.0):
        super().__init__()
        self.pad_frac = pad_frac
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)
        )
        self.bypass = nn.ModuleList(
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        x = self.lift(x)
        pad = 0
        if self.pad_frac > 0:
            pad = math.ceil(x.shape[-1] * self.pad_frac)
            x = F.pad(x, (0, pad, 0, pad))
        for spec, byp in zip(self.spectral, self.bypass):
            x = F.gelu(spec(x) + byp(x))
        if pad > 0:
            x = x[..., :-pad, :-pad]
        return self.proj(x)


def _conv_block(in_ch, out_ch, groups):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.GELU(),
    )


class UNet2d(nn.Module):
    """Standard 3-level U-Net; the strong conventional baseline.

    Downsampling gives it a global receptive field (unlike the plain CNN),
    but its kernels are still fixed in pixel units.
    """

    def __init__(self, in_channels=3, out_channels=1, base=36, groups=4):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = _conv_block(in_channels, c1, groups)
        self.enc2 = _conv_block(c1, c2, groups)
        self.enc3 = _conv_block(c2, c3, groups)
        self.bottleneck = _conv_block(c3, c4, groups)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _conv_block(c4, c3, groups)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _conv_block(c3, c2, groups)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _conv_block(c2, c1, groups)
        self.head = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class CNN2d(nn.Module):
    """Plain deep CNN, the 2-D analogue of the original project's baseline.

    Eight 3x3 conv layers -> receptive field of 17 pixels: it can only
    propagate information locally, which is the structural weakness this
    benchmark exposes on problems whose solution operator is global.
    """

    def __init__(self, in_channels=3, out_channels=1, width=208, depth=8, groups=8):
        super().__init__()
        layers = [nn.Conv2d(in_channels, width, 3, padding=1),
                  nn.GroupNorm(groups, width), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1),
                       nn.GroupNorm(groups, width), nn.GELU()]
        layers.append(nn.Conv2d(width, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    """Number of real scalar parameters (complex tensors count twice)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * (2 if p.is_complex() else 1)
    return total


def build_model(name, periodic=True, **overrides):
    """Model factory. `periodic` controls FNO domain padding only.

    Training-scheme variants share their base architecture: 'fnopf'
    (pushforward-trained FNO) builds the same network as 'fno'.
    """
    name = {"fnopf": "fno"}.get(name, name)
    if name == "fno":
        kwargs = {"modes": 12, "width": 32, "pad_frac": 0.0 if periodic else 0.125}
        kwargs.update(overrides)
        return FNO2d(**kwargs)
    if name == "unet":
        kwargs = {"base": 36}
        kwargs.update(overrides)
        return UNet2d(**kwargs)
    if name == "cnn":
        kwargs = {"width": 208}
        kwargs.update(overrides)
        return CNN2d(**kwargs)
    raise ValueError(f"unknown model '{name}'")


if __name__ == "__main__":
    for name in ("fno", "unet", "cnn"):
        model = build_model(name, periodic=False)
        print(f"{name:5s} {count_params(model):,} params")
