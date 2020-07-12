import torch.nn as nn
import torch.nn.functional as F
import torch

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class Biliteral_Grid(nn.Module):
    def __init__(self):
        super(Biliteral_Grid, self).__init__()
        self.SB1 = SplattingBlock(64,8,128) # 32 is not real
        self.SB2 = SplattingBlock(8, 16,256)
        self.SB3 = SplattingBlock(16, 32,512)
        self.conv1 = ConvLayer(32, 64,3,2)
        self.conv2 = ConvLayer(64, 64, 3, 1)

        # local feature
        self.L1 = ConvLayer(64, 64, 3, 1)
        self.L2 = ConvLayer(64, 64, 3, 1)

        # global feature
        self.G1 = ConvLayer(64, 64, 3, 2)
        self.G2 = ConvLayer(64, 64, 3, 2)
        self.G3 = nn.Linear(1024,256)
        self.G4 = nn.Linear(256,128)
        self.G5 = nn.Linear(128,64)
        self.G6 = nn.Linear(64,64)
        self.F = ConvLayer(128, 64, 1, 1)
        self.T = ConvLayer(64, 96, 3, 1)
        return

    def forward(self,c,s,feat):
        c,s = self.SB1(c,s,feat[0])
        c, s = self.SB2(c, s, feat[1])
        c, s = self.SB3(c, s, feat[2])

        c = F.relu(self.conv1(c))
        c = F.relu(self.conv2(c))

        # local feature
        L = F.relu(self.L1(c))
        L = F.relu(self.L2(L))

        # global feature
        G = F.relu(self.G1(c))
        G = F.relu(self.G2(G))
        G = G.reshape((G.shape[0],-1))
        G = F.relu(self.G3(G))
        G = F.relu(self.G4(G))
        G = F.relu(self.G5(G))
        G = F.relu(self.G6(G))

        G = G.reshape(G.shape+(1,1)).expand(G.shape+(16,16))
        f = torch.cat((L,G),dim=1)
        f = F.relu(self.F(f))
        f = self.T(f)
        # this is grid
        return f
#########################################################################################################
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Biliteral_Grid = Biliteral_Grid()
        self.guide = GuideNN()
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self,cont,cont_feat,style_feat):
        feat = []
        for i in range(1,len(cont_feat)):
            feat.append(adaptive_instance_normalization(cont_feat[i],style_feat[i]))

        coeffs_out = self.Biliteral_Grid(cont_feat[0],style_feat[0],feat)
        coeffs = coeffs_out.reshape(coeffs_out.shape[0],12,-1,coeffs_out.shape[-2],coeffs_out.shape[-1])
        guide = self.guide(cont)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, cont)
        # out -= out.min().detach()
        # out /= out.max().detach()
        return coeffs_out,out
########################################################################################################
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2 # same dimension after padding
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) # remember this dimension

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class SplattingBlock(nn.Module):
    def __init__(self,in_channels,out_channels,shortcut_channel):
        super(SplattingBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels,out_channels,kernel_size=3,stride=2)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.conv_short = nn.Conv2d(shortcut_channel, out_channels, 1, 1)
        return

    def forward(self,c,s,shortcut):
        c = F.relu(self.conv1(c))
        s = F.relu(self.conv1(s))
        c = adaptive_instance_normalization(c,s)
        shortcut = self.conv_short(shortcut)
        c += shortcut
        c = F.relu(self.conv2(c))
        return c,s

class LaplacianRegularizer(nn.Module):
    def __init__(self):
        super(LaplacianRegularizer, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
    def forward(self,f):
        loss = 0.
        for i in range(f.shape[2]):
            for j in range(f.shape[3]):
                up = max(i-1,0)
                down = min(i+1,f.shape[2] - 1)
                left = max(j-1,0)
                right = min(j+1,f.shape[3] - 1)
                term = f[:,:,i,j].view(f.shape[0],f.shape[1],1,1).expand(f.shape[0],f.shape[1],down - up+1,right-left+1)
                loss += self.mse_loss(term,f[:,:,up:down+1,left:right+1])
        return loss

# true laplacian_regularizer of original paper, input weight is coeffs in 5-dimension form
def calc_laplacian_regularizer_loss(self, weights, l1=0.0, l2=0.0):
        if not l1 and not l2:
            return 0.0
        diff1 = weights[:, :, 1:, :, :] - weights[:, :, :-1, :, :]
        diff2 = weights[:, :, :, 1:, :] - weights[:, :, :, :-1, :]
        diff3 = weights[:, :, :, :, 1:] - weights[:, :, :, :, :-1]
        if l1:
            result1 = torch.abs(diff1).sum()
            result1 += torch.abs(diff2).sum()
            result1 += torch.abs(diff3).sum()
        if l2:
            result2 = torch.pow(diff1, 2).sum()
            result2 += torch.pow(diff2, 2).sum()
            result2 += torch.pow(diff3, 2).sum()
        if l1 and not l2:
            return result1
        elif not l1 and l2:
            return result2
        else:
            return result1 + result2

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide)
        return coeff.squeeze(2)

class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)

class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=False)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x