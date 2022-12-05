import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import LossFunction


class Inception(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_size=(4,4,4)):
        super(Inception,self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1)
        )  
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels,hidden_size[0],kernel_size=1),
            nn.Conv2d(hidden_size[0], out_channels, kernel_size=5,padding=2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size[1], kernel_size=1),
            nn.Conv2d(hidden_size[1], hidden_size[2], kernel_size=3,padding=1),
            nn.Conv2d(hidden_size[2], out_channels, kernel_size=3,padding=1)
        )
        self.branch_pool = nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self, x):
        # print("Input shape: ", x.shape)
        branch1 = self.branch1(x)
        branch5 = self.branch5(x)
        branch3 = self.branch3(x)
        branch_pool = F.avg_pool2d(x, kernel_size=3,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)
        # print("Output shape: ", branch1.shape, branch5.shape, branch3.shape, branch_pool.shape)

        out = torch.cat(
            (branch1.unsqueeze(0), 
            branch5.unsqueeze(0), 
            branch3.unsqueeze(0), 
            branch_pool.unsqueeze(0)), 
            dim=0)
        out = torch.mean(out, dim=0)
        return out
        # return torch.cat((branch1,branch5,branch3,branch_pool),dim=1)
        

class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu

class EnhanceInceptionNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceInceptionNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            Inception(in_channels=3,out_channels=channels),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            Inception(in_channels=channels,out_channels=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            # nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            Inception(in_channels=channels,out_channels=3),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu
class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta

class CalibrateInceptionNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateInceptionNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            Inception(in_channels=3,out_channels=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            # nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            Inception(in_channels=channels,out_channels=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            # nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            Inception(in_channels=channels,out_channels=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta

class Network(nn.Module):

    def __init__(self, stage=3,inception=False):
        super(Network, self).__init__()
        self.stage = stage
        if inception:
            self.enhance = EnhanceInceptionNetwork(layers=stage, channels=32)
            self.calibrate = CalibrateInceptionNetwork(layers=stage, channels=32)
        else:
            self.enhance = EnhanceNetwork(layers=1, channels=3)
            self.calibrate = CalibrateNetwork(layers=3, channels=16)
        self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.enhance(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            att = self.calibrate(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss



class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self._criterion = LossFunction()

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss

