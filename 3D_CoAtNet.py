import torch 
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from collections import OrderedDict

def conv_3x3_bn_3d(in_c, out_c, image_size, downsample=False):
    stride = (2, 2, 2) if downsample else(1, 1, 1)
    layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(out_c),
        nn.GELU()
    )
    return layer



class SE3D(nn.Module):
    def __init__(self, in_c, out_c, expansion=0.25):
        super(SE3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_c, int(in_c * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(in_c * expansion), out_c, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        y = self.fc(y).view(b, c, 1, 1, 1)
        
        return x * y 




class MBConv3D(nn.Module):
    def __init__(self, in_c, out_c, image_size, downsample=False, expansion=4):
        super(MBConv3D, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        hidden_dim = int(in_c * expansion)

        if self.downsample:
            self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv3d(in_c, out_c, 1, 1, 0, bias=False)

        layers = OrderedDict()
        # Expand
        expand_conv = nn.Sequential(
            nn.Conv3d(in_c, hidden_dim, 1, stride, 0, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"expand_conv": expand_conv})

        # Depthwise Conv
        dw_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"dw_conv": dw_conv})

        # SE block adapted for 3D
        layers.update({"se": SE3D(in_c, hidden_dim)})

        # Project
        pro_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_c)
        )
        layers.update({"pro_conv": pro_conv})
        self.block = nn.Sequential(layers)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.block(x)
        else:
            return x + self.block(x)



class FFN3D(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5, kernel_size=1):
        super(FFN3D, self).__init__()
        # Use 1x1x1 convolutions to emulate fully connected behavior on 3D data
        self.ffn = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size, stride=1, padding=0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden_channels, in_channels, kernel_size, stride=1, padding=0),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)




class AttentionD(nn.Module):
    def __init__(self, inp, oup, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, d, h, w):
        # Dynamically calculate relative position bias based on input dimensions
        relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * d - 1) * (2 * h - 1) * (2 * w - 1), self.heads)
        )

        coords_d = torch.arange(d)
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        #print("coords shape", coords.shape)

        coords_flatten = torch.flatten(coords, 1)
        #print("coords_flatten shape afetr flatten ", coords_flatten.shape)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += d - 1
        relative_coords[:, :, 1] += h - 1
        relative_coords[:, :, 2] += w - 1

        relative_coords[:, :, 0] *= (2 * h - 1) * (2 * w - 1)
        relative_coords[:, :, 1] *= (2 * w - 1)
        relative_position_index = relative_coords.sum(-1)
        #print("relative_position_index after sum shape", relative_position_index.shape)


        # Continue with forward pass as before, using the dynamically calculated relative position bias
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        #print("q shape", q.shape)
        #print("k shape", k.shape)
        #print("v shape", v.shape)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #print("dots shape", dots.shape)

        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(
            -1, d * h * w, self.heads)
        #print("relative position bias shape", relative_position_bias.shape)
     
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().to(x.device) 
        #print("relative position bias shape after permute", relative_position_bias.shape)

        dots = dots + relative_position_bias
        #print("dots shape", dots.shape)


        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        #print("out shaop", out.shape)

        return out





class Transformer(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 image_size,
                 heads=8,
                 dim_head=32,
                 downsample=False,
                 dropout=0.5,
                 expansion=4,
                 layer_norm = nn.LayerNorm,
                batch_norm = nn.BatchNorm3d):
        super(Transformer, self).__init__()
        self.downsample = downsample
        hidden_dim = int(in_c * expansion)
        self.d, self.h, self.w = image_size

        if self.downsample:
            self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv3d(in_c, out_c, 1, 1, 0, bias=False)

        self.attn = AttentionD(in_c, out_c, heads, dim_head, dropout)
        self.ffn = FFN3D(out_c, hidden_dim, dropout) 
        self.layer_norm = layer_norm(in_c)
        self.batch_norm = batch_norm(out_c)

    def forward(self, x):
        if self.downsample:
            x1 = self.pool1(x)
            x2 = self.pool2(x)
            x2 = self.proj(x2)
        else:
            x1 = x
            x2 = x

        d, h, w = x1.shape[-3:]
        x1 = rearrange(x1, 'b c d h w -> b (d h w) c')
        #print("x1", x1.shape)
        x1 = self.layer_norm(x1)
        x1 = self.attn(x1, d, h, w)
        x1 = rearrange(x1, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

        x3 = x1 + x2
        x3 = self.batch_norm(x3)
        x3 = self.ffn(x3)  

        out = x1 + x3
        #print("out, x1. + x3", out.shape)

        return out



class CoAtNet(nn.Module):
    def __init__(self,
                 image_size=(40, 128, 128),
                 in_channels: int = 3,
                 num_blocks: list = [1, 1, 1, 1, 1],  # L
                 channels: list = [16, 16, 16, 16, 16],  # D
                 num_classes: int = 2,
                 block_types=['C', 'C', 'C', 'T']):
        super(CoAtNet, self).__init__()

        #assert len(image_size) == 2, "image size must be: {H,W}"
        assert len(channels) == 5
        assert len(block_types) == 4

        id, ih, iw = image_size
        block = {'C': MBConv3D, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn_3d, in_channels, channels[0], num_blocks[0], (id, ih, iw), 
        )
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (id, ih, iw)
        )
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (id, ih , iw)
        )
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (id, ih, iw)
        )
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (id, ih, iw)
        )

        # 
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], num_classes, bias=False),
            nn.Softmax(dim=1)
        )
        

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print("input x shape", x.shape)
        x = self.s0(x)
        #print("input x shape after s0", x.shape)

        x = self.s1(x)
        #print("input x shape after s1", x.shape)

        x = self.s2(x)
        #print("input x shape after s2", x.shape)

        x = self.s3(x)
        #print("input x shape after s3", x.shape)

        x = self.s4(x)
        #print("input x shape after s4", x.shape)


        x = self.pool(x)
        #print("input x shape after pooling", x.shape)

        x = torch.flatten(x, 1)
        #print("input x shape after pooling", x.shape)

        x = self.fc(x)
        #print("input x shape after FC", x.shape)

        return x
    def _make_layer(self, block, in_c, out_c, depth, image_size, downsample=True):
        layers = nn.ModuleList([])
        for i in range(depth):
            layers.append(block(in_c if i == 0 else out_c, out_c, image_size, downsample=(i == 0 and downsample)))
        return nn.Sequential(*layers)
    




# Example

def coatnet_0(num_classes=2):
    num_blocks = [1, 1, 1, 2, 1]  # L
    channels = [32, 32, 32, 32, 32]  # D
    return CoAtNet((10, 50, 50), 1, num_blocks, channels, num_classes=num_classes)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = coatnet_0().to(device)
img = torch.randn(10, 1, 50, 128, 128).to(device)
out = model(img)
summary(model, input_size=(10, 1, 50, 128, 128))




