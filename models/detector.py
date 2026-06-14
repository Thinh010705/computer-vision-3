import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Sequential):
    """Tinh chỉnh đặc trưng FPN sau khi dung hợp bằng Conv-BN-SiLU."""

    def __init__(self, in_channels, out_channels):
        """Tạo khối tích chập chuẩn để tinh chỉnh đặc trưng đã dung hợp."""
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableConvBlock(nn.Sequential):
    """Tích chập tách theo chiều sâu giúp đầu phát hiện nhẹ và ít tham số hơn."""

    def __init__(self, in_channels, out_channels):
        """Tạo depthwise convolution nối với pointwise convolution."""
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DecoupledHead(nn.Module):
    """Tách riêng nhánh phân lớp/objectness và nhánh hồi quy hộp bao."""

    def __init__(self, in_channels=256, hidden_channels=160, dropout=0.15):
        """Tạo hai nhánh dự đoán độc lập cho phân lớp và hồi quy."""
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
        """Ghép kết quả hai nhánh thành tensor dự đoán 10 kênh."""
        # Cấu trúc đầu ra: [objectness, 5 class logits, tx, ty, tw, th].
        return torch.cat((self.cls_head(x), self.reg_head(x)), dim=1)


class ConvNeXtFPNDetector(nn.Module):
    """
    Bộ phát hiện anchor-free đa tỉ lệ dùng ConvNeXt-Small và FPN P3/P4/P5.

    Backbone cung cấp đặc trưng tại stride 8, 16 và 32. FPN truyền thông tin
    ngữ nghĩa từ tầng sâu xuống các tầng có độ phân giải cao hơn, sau đó mỗi
    tầng sử dụng một đầu dự đoán độc lập.
    """

    strides = (8, 16, 32)
    model_version = "convnext_fpn_p3p5_v2"

    def __init__(self, pretrained=True):
        """Khởi tạo ConvNeXt-Small và FPN top-down."""
        super().__init__()
        self.backbone_features = self._build_backbone(pretrained).features

        self.proj8 = nn.Conv2d(192, 256, kernel_size=1, bias=False)
        self.proj16 = nn.Conv2d(384, 256, kernel_size=1, bias=False)
        self.proj32 = nn.Conv2d(768, 256, kernel_size=1, bias=False)

        self.fuse16 = ConvBlock(512, 256)
        self.fuse8 = ConvBlock(512, 256)
        self.refine32 = ConvBlock(256, 256)

        self.head8 = DecoupledHead()
        self.head16 = DecoupledHead()
        self.head32 = DecoupledHead()

    @staticmethod
    def _build_backbone(pretrained):
        """Tạo ConvNeXt-Small tương thích với cả API torchvision cũ và mới."""
        try:
            if pretrained:
                from torchvision.models import ConvNeXt_Small_Weights
                weights = ConvNeXt_Small_Weights.DEFAULT
            else:
                weights = None
            return models.convnext_small(weights=weights)
        except (ImportError, TypeError):
            return models.convnext_small(pretrained=pretrained)

    def forward(self, x):
        """Trả về ba tensor dự đoán tương ứng với stride 8, 16 và 32."""
        # Các vị trí trong ConvNeXt cung cấp c2/c3/c4 tại stride 8/16/32.
        x = self.backbone_features[0](x)
        x = self.backbone_features[1](x)
        x = self.backbone_features[2](x)
        c2 = self.backbone_features[3](x)
        x = self.backbone_features[4](c2)
        c3 = self.backbone_features[5](x)
        x = self.backbone_features[6](c3)
        c4 = self.backbone_features[7](x)

        # FPN top-down truyền thông tin ngữ nghĩa sâu xuống các lưới mịn hơn.
        p32 = self.refine32(self.proj32(c4))
        p16 = self.fuse16(torch.cat((self.proj16(c3), F.interpolate(p32, size=c3.shape[-2:], mode="bilinear", align_corners=False)), dim=1))
        p8 = self.fuse8(torch.cat((self.proj8(c2), F.interpolate(p16, size=c2.shape[-2:], mode="bilinear", align_corners=False)), dim=1))

        return [self.head8(p8), self.head16(p16), self.head32(p32)]
