import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DecoupledHead(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=160, dropout=0.15):
        super().__init__()
        self.cls_head = nn.Sequential(
            SeparableConvBlock(in_channels, hidden_channels),
            SeparableConvBlock(hidden_channels, hidden_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, 6, kernel_size=1),
        )
        self.reg_head = nn.Sequential(
            SeparableConvBlock(in_channels, hidden_channels),
            SeparableConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 4, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat((self.cls_head(x), self.reg_head(x)), dim=1)


class ConvNeXtFPNDetector(nn.Module):
    """
    Anchor-free multi-scale detector using a ConvNeXt backbone and P3/P4/P5 FPN.

    ConvNeXt-Tiny and ConvNeXt-Small both expose stride-8/16/32 features with
    192/384/768 channels.
    """

    strides = (8, 16, 32)
    model_version = "convnext_fpn_pan_p3p5_v3"

    def __init__(self, pretrained=True, backbone_name="tiny", neck_name="fpn"):
        super().__init__()
        if backbone_name not in {"tiny", "small"}:
            raise ValueError("backbone_name must be 'tiny' or 'small'")
        if neck_name not in {"fpn", "pan"}:
            raise ValueError("neck_name must be 'fpn' or 'pan'")
        self.backbone_name = backbone_name
        self.neck_name = neck_name
        self.backbone_features = self._build_backbone(backbone_name, pretrained).features

        self.proj8 = nn.Conv2d(192, 256, kernel_size=1, bias=False)
        self.proj16 = nn.Conv2d(384, 256, kernel_size=1, bias=False)
        self.proj32 = nn.Conv2d(768, 256, kernel_size=1, bias=False)

        self.fuse16 = ConvBlock(512, 256)
        self.fuse8 = ConvBlock(512, 256)
        self.refine32 = ConvBlock(256, 256)

        if neck_name == "pan":
            # Bottom-up path aggregation sends precise P3 localization features
            # back to P4/P5 after the normal top-down FPN fusion.
            self.down8 = SeparableConvBlock(256, 256, stride=2)
            self.pan16 = SeparableConvBlock(512, 256)
            self.down16 = SeparableConvBlock(256, 256, stride=2)
            self.pan32 = SeparableConvBlock(512, 256)

        self.head8 = DecoupledHead()
        self.head16 = DecoupledHead()
        self.head32 = DecoupledHead()

    @staticmethod
    def _build_backbone(backbone_name, pretrained):
        constructor = models.convnext_tiny if backbone_name == "tiny" else models.convnext_small
        try:
            if pretrained:
                if backbone_name == "tiny":
                    from torchvision.models import ConvNeXt_Tiny_Weights
                    weights = ConvNeXt_Tiny_Weights.DEFAULT
                else:
                    from torchvision.models import ConvNeXt_Small_Weights
                    weights = ConvNeXt_Small_Weights.DEFAULT
            else:
                weights = None
            return constructor(weights=weights)
        except (ImportError, TypeError):
            return constructor(pretrained=pretrained)

    def forward(self, x):
        # ConvNeXt feature indices: c2=stride 8, c3=stride 16, c4=stride 32.
        x = self.backbone_features[0](x)
        x = self.backbone_features[1](x)
        x = self.backbone_features[2](x)
        c2 = self.backbone_features[3](x)
        x = self.backbone_features[4](c2)
        c3 = self.backbone_features[5](x)
        x = self.backbone_features[6](c3)
        c4 = self.backbone_features[7](x)

        p32 = self.refine32(self.proj32(c4))
        p16 = self.fuse16(torch.cat((self.proj16(c3), F.interpolate(p32, size=c3.shape[-2:], mode="bilinear", align_corners=False)), dim=1))
        p8 = self.fuse8(torch.cat((self.proj8(c2), F.interpolate(p16, size=c2.shape[-2:], mode="bilinear", align_corners=False)), dim=1))

        if self.neck_name == "pan":
            p16 = self.pan16(torch.cat((p16, self.down8(p8)), dim=1))
            p32 = self.pan32(torch.cat((p32, self.down16(p16)), dim=1))

        return [self.head8(p8), self.head16(p16), self.head32(p32)]
