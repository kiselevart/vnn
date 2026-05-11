import torch
import torch.nn as nn
import torchvision.models as models

from network.cifar.vnn_cifar import VNN_CIFAR
from network.cifar_ortho.res_vnn_ortho import ResVNN_Ortho_CIFAR

# Legacy video models (vnn_rgb, vnn_fusion)
from network.video import (
    vnn_rgb_of_highQv2,
)
from network.video import vnn_fusion_highQ, vnn_rgb_of_highQ
from network.video.established_models import R2Plus1DNet, R3DNet

# Higher-order video models
from network.video_higher_order import (
    VNNRgbHO, VNNFusionHO,
    lvn_rgb_gauss, lvn_rgb_signed,
    lvn_fusion_gauss, lvn_fusion_signed,
    lvn_laguerre_rgb, lvn_laguerre_fusion,
    lvn_monomial_rgb, lvn_monomial_fusion,
    lvn_laguerre_full_rgb, lvn_laguerre_full_fusion,
)

from network.video import vnn_fusion_highQv2
from network.timeseries import (
    VNN1D, LaguerreVNN1D,
    LaguerreVNN1D_S1, LaguerreVNN1D_S2, LaguerreVNN1D_S3,
    LaguerreVNN1D_S4, LaguerreVNN1D_S5,
    LaguerreVNN1D_S6, LaguerreVNN1D_S7, LaguerreVNN1D_S8,
)


def get_model(args, device):
    print(f"==> Building model: {args.model}")

    if args.task == "cifar":
        if args.model == "vnn_simple":
            net = VNN_CIFAR(num_classes=args.num_classes)
        elif args.model == "vnn_ortho":
            # Default to [2,2,2,2] blocks for ResNet18 equivalence
            net = ResVNN_Ortho_CIFAR(
                num_classes=args.num_classes, num_blocks=[2, 2, 2, 2], Q=args.Q
            )
        elif args.model == "resnet18":
            net = models.resnet18(weights=None)
            net.fc = nn.Linear(net.fc.in_features, args.num_classes)
            # Adapt for CIFAR-10 size
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        else:
            raise ValueError(f"Unknown CIFAR model: {args.model}")

    elif args.task == "video":
        clip_len = getattr(args, "clip_len", 16)
        if args.model == "vnn_rgb":
            # Legacy: RGB Backbone -> Classifier
            class VideoVNN(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.backbone = vnn_rgb_of_highQv2.VNN(
                        num_classes=num_classes, num_ch=3, pretrained=False
                    )
                    self.head = vnn_fusion_highQv2.VNN_F(
                        num_classes=num_classes, num_ch=96, pretrained=False
                    )

                def forward(self, x):
                    feats = self.backbone(x)
                    return self.head(feats)

                def get_1x_lr_params(self):
                    return list(
                        vnn_rgb_of_highQv2.get_1x_lr_params(self.backbone)
                    ) + list(self.head.parameters())

            net = VideoVNN(num_classes=args.num_classes)

        elif args.model == "vnn_fusion":
            # Legacy: RGB + Flow streams
            class VideoVNNFusion(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.model_rgb = vnn_rgb_of_highQv2.VNN(
                        num_classes=num_classes, num_ch=3, pretrained=False
                    )
                    self.model_of = vnn_rgb_of_highQv2.VNN(
                        num_classes=num_classes, num_ch=2, pretrained=False
                    )
                    self.model_fuse = vnn_fusion_highQv2.VNN_F(
                        num_classes=num_classes, num_ch=192, pretrained=False
                    )

                def forward(self, x):
                    rgb, flow = x
                    out_rgb = self.model_rgb(rgb)
                    out_of = self.model_of(flow)
                    out_fuse = self.model_fuse(torch.cat((out_rgb, out_of), 1))
                    return out_fuse

                def get_1x_lr_params(self):
                    p = []
                    p += list(vnn_rgb_of_highQv2.get_1x_lr_params(self.model_rgb))
                    p += list(vnn_rgb_of_highQv2.get_1x_lr_params(self.model_of))
                    p += list(vnn_fusion_highQv2.get_1x_lr_params(self.model_fuse))
                    return p

                def get_10x_lr_params(self):
                    return list(vnn_fusion_highQv2.get_10x_lr_params(self.model_fuse))

            net = VideoVNNFusion(num_classes=args.num_classes)

        elif args.model == "vnn_fusion_orig":
            class VideoVNNFusionOrig(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.model_rgb = vnn_rgb_of_highQ.VNN(
                        num_classes=num_classes, num_ch=3, pretrained=False
                    )
                    self.model_of = vnn_rgb_of_highQ.VNN(
                        num_classes=num_classes, num_ch=2, pretrained=False
                    )
                    self.model_fuse = vnn_fusion_highQ.VNN_F(
                        num_classes=num_classes, num_ch=192, pretrained=False
                    )

                def forward(self, x):
                    rgb, flow = x
                    out_rgb = self.model_rgb(rgb)
                    out_of = self.model_of(flow)
                    return self.model_fuse(torch.cat((out_rgb, out_of), 1))

            net = VideoVNNFusionOrig(num_classes=args.num_classes)

        elif args.model == "vnn_rgb_ho":
            net = VNNRgbHO(num_classes=args.num_classes, cubic_mode=args.cubic_mode,
                           use_cubic=not args.disable_cubic, clip_len=clip_len)

        elif args.model == "vnn_fusion_ho":
            net = VNNFusionHO(num_classes=args.num_classes, cubic_mode=args.cubic_mode,
                              use_cubic=not args.disable_cubic, clip_len=clip_len)

        # --- Laguerre VNN ablations ---
        elif args.model == "lvn_rgb_gauss":
            net = lvn_rgb_gauss(num_classes=args.num_classes, clip_len=clip_len)

        elif args.model == "lvn_rgb_signed":
            net = lvn_rgb_signed(num_classes=args.num_classes, clip_len=clip_len)

        elif args.model == "lvn_fusion_gauss":
            net = lvn_fusion_gauss(num_classes=args.num_classes, clip_len=clip_len)

        elif args.model == "lvn_fusion_signed":
            net = lvn_fusion_signed(num_classes=args.num_classes, clip_len=clip_len)

        elif args.model == "lvn_laguerre_rgb":
            n_lag = getattr(args, "n_lag", None)
            net = lvn_laguerre_rgb(num_classes=args.num_classes, clip_len=clip_len, n_lag=n_lag)

        elif args.model == "lvn_laguerre_fusion":
            n_lag = getattr(args, "n_lag", None)
            net = lvn_laguerre_fusion(num_classes=args.num_classes, clip_len=clip_len, n_lag=n_lag)

        elif args.model == "lvn_monomial_rgb":
            net = lvn_monomial_rgb(num_classes=args.num_classes, clip_len=clip_len)

        elif args.model == "lvn_monomial_fusion":
            net = lvn_monomial_fusion(num_classes=args.num_classes, clip_len=clip_len)

        elif args.model == "lvn_laguerre_full_rgb":
            net = lvn_laguerre_full_rgb(
                num_classes=args.num_classes, clip_len=clip_len,
                n_lag_t=getattr(args, "n_lag_t", None),
                n_lag_s=getattr(args, "n_lag_s", None),
            )

        elif args.model == "lvn_laguerre_full_fusion":
            net = lvn_laguerre_full_fusion(
                num_classes=args.num_classes, clip_len=clip_len,
                n_lag_t=getattr(args, "n_lag_t", None),
                n_lag_s=getattr(args, "n_lag_s", None),
            )

        elif args.model == "r2plus1d":
            net = R2Plus1DNet(num_classes=args.num_classes)

        elif args.model == "r3d":
            net = R3DNet(num_classes=args.num_classes)

        else:
            raise ValueError(f"Unknown Video model: {args.model}")

    elif args.task == "timeseries":
        in_ch = getattr(args, "in_ch", 1)
        if args.model == "vnn_1d":
            net = VNN1D(
                num_classes=args.num_classes,
                in_ch=in_ch,
                base_ch=getattr(args, "base_ch", 8),
                Q=getattr(args, "Q", 2),
                Qc=getattr(args, "Qc", 1),
                cubic_mode=getattr(args, "cubic_mode", "symmetric"),
                use_cubic=not getattr(args, "disable_cubic", False),
            )
        elif args.model == "laguerre_vnn_1d":
            net = LaguerreVNN1D(
                num_classes=args.num_classes,
                in_ch=in_ch,
                base_ch=getattr(args, "base_ch", 8),
                poly_degrees=getattr(args, "poly_degrees", None),
                alpha=getattr(args, "alpha", 1.0),
            )
        elif args.model in ("laguerre_vnn_1d_s1", "laguerre_vnn_1d_s2", "laguerre_vnn_1d_s3",
                            "laguerre_vnn_1d_s4", "laguerre_vnn_1d_s5",
                            "laguerre_vnn_1d_s6", "laguerre_vnn_1d_s7", "laguerre_vnn_1d_s8"):
            _cls = {"laguerre_vnn_1d_s1": LaguerreVNN1D_S1,
                    "laguerre_vnn_1d_s2": LaguerreVNN1D_S2,
                    "laguerre_vnn_1d_s3": LaguerreVNN1D_S3,
                    "laguerre_vnn_1d_s4": LaguerreVNN1D_S4,
                    "laguerre_vnn_1d_s5": LaguerreVNN1D_S5,
                    "laguerre_vnn_1d_s6": LaguerreVNN1D_S6,
                    "laguerre_vnn_1d_s7": LaguerreVNN1D_S7,
                    "laguerre_vnn_1d_s8": LaguerreVNN1D_S8}[args.model]
            net = _cls(
                num_classes=args.num_classes,
                in_ch=in_ch,
                base_ch=getattr(args, "base_ch", 8),
                poly_degrees=getattr(args, "poly_degrees", None),
                alpha=getattr(args, "alpha", 1.0),
            )
        elif args.model in ("fcn", "resnet1d", "inceptiontime"):
            try:
                from tsai.models.FCN import FCN
                from tsai.models.ResNet import ResNet
                from tsai.models.InceptionTime import InceptionTime
            except ImportError as e:
                raise ImportError("Install tsai for baseline models:  pip install tsai") from e
            _tsai = {"fcn": FCN, "resnet1d": ResNet, "inceptiontime": InceptionTime}
            net = _tsai[args.model](c_in=in_ch, c_out=args.num_classes)
        else:
            raise ValueError(f"Unknown timeseries model: {args.model}")

    elif args.task == "mnist":
        from network.mnist import TinyCNN, TinyVNN, TinyLaguerreVNN
        nc = getattr(args, "num_classes", 10)
        ch = getattr(args, "base_ch", 8)
        if args.model == "tiny_cnn":
            net = TinyCNN(num_classes=nc, base_ch=ch)
        elif args.model == "tiny_vnn":
            net = TinyVNN(num_classes=nc, base_ch=ch)
        elif args.model == "tiny_laguerre":
            net = TinyLaguerreVNN(
                num_classes=nc,
                base_ch=ch,
                poly_degrees=getattr(args, "poly_degrees", [2, 3]),
                alpha=getattr(args, "alpha", 0.5),
            )
        else:
            raise ValueError(f"Unknown MNIST model: {args.model}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    return net.to(device)
