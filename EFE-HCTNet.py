import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary
from torchvision.models import ResNet34_Weights
torch.cuda.set_device(0)
from einops.einops import rearrange

# --------------------------------------------------- ViT --------------------------------------------------------
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
# ----------------------------------------------------- ASPP -----------------------------------------------------------
class ASPP(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPP, self).__init__()

        dilations = [1, 3, 5,7]

        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,padding=0, dilation=dilations[0], bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU())
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=dilations[1], dilation=dilations[1], bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU())
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=dilations[2], dilation=dilations[2], bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU())
        self.aspp4 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=dilations[3], dilation=dilations[3], bias=False),
                                                 nn.BatchNorm2d(planes),
                                                 nn.ReLU())
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(planes),
                                             nn.ReLU())
        self.global_max_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(planes),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(planes*5, inplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.global_avg_pool(x)
        Y4 = self.global_max_pool(x)
        x4 = x4 + Y4
        x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x5 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4,x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# -----------------  GLFRM --------------------------------------------------------------
class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                      ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

# ------------------------------------------------------------ MSF -----------------------------------------------------
class MSF(nn.Module):
    def __init__(self, c1, c2):
        super(MSF, self).__init__()
        self.dconv1 = nn.Conv2d(in_channels=c1, out_channels=c1 // 2, kernel_size=3, dilation=2, padding=2)
        self.dconv2 = nn.Conv2d(in_channels=c1 // 4, out_channels=c1 // 4, kernel_size=3, dilation=4, padding=4)
        self.dconv3 = nn.Conv2d(in_channels=c1 // 4, out_channels=c1 // 4, kernel_size=3, dilation=8, padding=8)
        self.conv1 = nn.Conv2d(in_channels=3 * c1 // 2, out_channels=c1 // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=7 * c1 // 4, out_channels=c1 // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=2 * c1, out_channels=c2, kernel_size=1)

    def forward(self, x):
        out = torch.cat((x, self.dconv1(x)), dim=1)
        out = torch.cat((out, self.dconv2(self.conv1(out))), dim=1)
        out = torch.cat((out, self.dconv3(self.conv2(out))), dim=1)
        out = self.conv3(out)
        return out


# --------------------------------------------------------- CSAM -------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):

        z1 = self.avg_pool(z)
        z2 = self.max_pool(z)
        z = z1 + z2
        z = z.squeeze(-1)
        z = self.conv1d(z)
        z = self.sigmoid(z.unsqueeze(-1))
        return z

class SpatialAttention(nn.Module):
    def __init__(self, in_planes, H, W):
        super(SpatialAttention, self).__init__()
        self.avg_pooled_w = nn.AvgPool2d((H, 1))
        self.max_pooled_w = nn.MaxPool2d((H, 1))

        self.avg_pooled_h = nn.AvgPool2d((1, W))
        self.max_pooled_h = nn.MaxPool2d((1, W))

        self.conv = nn.Conv2d(in_planes, in_planes, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled_w = self.avg_pooled_w(x)
        max_pooled_w = self.max_pooled_w(x)
        feature_map_w = avg_pooled_w + max_pooled_w

        avg_pooled_h = self.avg_pooled_h(x)
        max_pooled_h = self.max_pooled_h(x)
        feature_map_h = avg_pooled_h + max_pooled_h

        feature_map_w_conv = self.conv(feature_map_w)
        feature_map_h_conv = self.conv(feature_map_h)

        feature_map_w_sig = self.sigmoid(feature_map_w_conv)
        feature_map_h_sig = self.sigmoid(feature_map_h_conv)

        reconstructed_feature_map = torch.matmul(feature_map_h_sig, feature_map_w_sig)

        return reconstructed_feature_map


# ------------------------ M ----------------------

class CSAM(nn.Module):
    def __init__(self, in_planes, H, W):
        super(CSAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention(in_planes, H, W)
        self.MSF = MSF(in_planes, in_planes)

    def forward(self, x,y):
        z = x + y
        z = self.MSF(z)
        z1 = y * self.ca(z)
        z2 = y * self.sa(z)
        z = z1 + z2
        out = torch.cat([x, z], dim=1)

        return out


# 上采样
class Decoder(nn.Module):
    def __init__(self, channels,channels1, channels2, H, W):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout(0.05),
                nn.Conv2d(channels, channels1, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels1),
                nn.ReLU(inplace=True)
            )

        self.upconv = nn.Sequential(
                nn.Upsample(size=(H, W), mode='bilinear', align_corners=False),
                nn.Conv2d(channels1, channels2, 1, 1, 0),
                nn.BatchNorm2d(channels2),
                nn.ReLU(inplace=True)
            )
        self.csa = CSAM(channels2, H, W)

    def forward(self, x, y):
        x = self.conv(x)
        x = self.upconv(x)
        x = self.csa(x, y)
        return x

# ---------------------------------------------------------- EFSM ------------------------------------------------------
class Curvature(torch.nn.Module):
    def __init__(self, ratio):
        super(Curvature, self).__init__()
        weights = torch.tensor([[[[-1 / 16, 5 / 16, -1 / 16], [5 / 16, -1, 5 / 16], [-1 / 16, 5 / 16, -1 / 16]]]])
        self.weight = torch.nn.Parameter(weights)  # .to(device)
        self.ratio = ratio

    def forward(self, x):
        B, C, H, W = x.size()
        x_origin = x
        x = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x, self.weight)
        out = torch.abs(out)
        p = torch.sum(out, dim=-1)
        p = torch.sum(p, dim=-1)
        p = p.reshape(B, C)

        _, index = torch.topk(p, int(self.ratio), dim=1)
        selected = []
        for i in range(x_origin.shape[0]):
            selected.append(torch.index_select(x_origin[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)

        return selected


# ----------------------------------------------- Segmentation head ----------------------------------------------------
class Seg_head(nn.Module):
    def __init__(self):
        super(Seg_head, self).__init__()

        self.up1 = nn.Upsample(size=(256, 320), mode='bilinear', align_corners=False)
        self.up2 = nn.Sequential(
                nn.Conv2d(128, 128, 1, 1, 0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(256, 320), mode='bilinear', align_corners=False)
            )
        self.EFSM = Curvature(64)
        self.conv0 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False),
                     nn.BatchNorm2d(64),
                     nn.ReLU(inplace=True),
                    )
        self.edge = nn.Conv2d(64, 2, 1, bias=False)

        self.conv1 = nn.Conv2d(192, 96, 1, bias=False)
        self.conv2 = nn.Conv2d(96, 2, 1, bias=False)


    def forward(self, x1,x2,x3):
        x1 = self.up1(x1)
        x2 = self.up1(x2)

        x3 = self.up2(x3)
        x = torch.cat([x1, x2], dim=1)

        x1 = self.EFSM(x)
        x1= self.conv0(x1)
        edge_logits = self.edge(x1)

        x = torch.cat([x1, x3], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x, edge_logits


# ---------------------------------------------- Network -------------------------------------------------------

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # ------------------------------------------ Encode ---------------------------------------
        base = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        #base = torchvision.models.resnet34(pretrained = False)
        self.encoder1 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu
        )

        self.encoder2 = nn.Sequential(base.maxpool, base.layer1)
        self.encoder3 = base.layer2
        self.encoder4 = base.layer3
        self.VITBlock = MobileViTBlock(256, 3, 256, 3, (2, 2), int(256 * 2))
        self.ASPP = ASPP(256, 128)

        # -------------------------------------------- Dncode -------------------------------------
        self.decoder1 = Decoder(256, 128,128, 32, 40)  # (128,32,40)
        self.decoder2 = Decoder(256, 128, 64, 64, 80)  # (64,64,80)
        self.decoder3 = Decoder(128, 64, 64, 128, 160)  # (32,128,160)
        self.seg_head = Seg_head()


    def forward(self, x):

    # ------ Encoder ------
        d1 = self.encoder1(x)
        d2 = self.encoder2(d1)
        d3 = self.encoder3(d2)
        #print('d3: ', d3.size())
        d4 = self.encoder4(d3)
    # --------------------- GLFRM --------------------
        A = self.ASPP(d4)
        V = self.VITBlock(A)
        V = V + d4
    # --------------------- Decoder -----------------
        u1 = self.decoder1(V, d3)
        u2 = self.decoder2(u1, d2)
        u3 = self.decoder3(u2, d1)
    # -------- Segmentation head --------
        S_g_pred, edge_logits = self.seg_head(d1,d2,u3)
        return S_g_pred, edge_logits

if __name__ == "__main__":
    model = Network()
    model.to('cuda:0')
    summary(model, (3, 256, 320))




































