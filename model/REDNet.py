'''
REDNet with 15 layers
Reference :
"Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections"
"
'''
import torch
from torch import nn
import math

def he_initialization(m):
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

class encoder_block(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True, weight_initialization=True):
        super(encoder_block,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

        if weight_initialization:
            he_initialization(self.conv)

        if activation:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self,x):
        x = self.conv(x)
        if self.activation:
            out = self.activation(x)
        else:
            out = x
        return out

class decoder_block(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3,stride=1, padding=1, activation=True, weight_initialization=True):
        super(decoder_block,self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

        if weight_initialization:
            he_initialization(self.deconv)

        if activation:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self,x):
        x = self.deconv(x)
        if self.activation:
            out = self.activation(x)
        else:
            out = x
        return out


class REDNet15(nn.Module):
    def __init__(self, num_blocks=15, num_features=64, input_channel = 1,output_channel = 1):
        super(REDNet15,self).__init__()

        self.num_blocks = num_blocks
        encoders=[]
        decoders=[]

        encoders.append(encoder_block(input_channel,num_features))

        for i in range(num_blocks-1):
            encoders.append(encoder_block(num_features,num_features))
        for i in range(num_blocks-1):
            decoders.append(decoder_block(num_features,num_features))

        decoders.append(decoder_block(num_features,output_channel))

        self.encoders = nn.Sequential(*encoders)
        self.decoders = nn.Sequential(*decoders)
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        input = x
        residuals = []
        for i in range(self.num_blocks):
            x = self.encoders[i](x)
            if (i+1)%2 == 0 and len(residuals)<math.ceil(self.num_blocks/2)-1:
                residuals.append(x)


        residuals_idx = 0
        for i in range(self.num_blocks):
            x = self.decoders[i](x)
            if (i+1+self.num_blocks) % 2 == 0 and residuals_idx < len(residuals):
                res = residuals[-(residuals_idx+1)]
                residuals_idx += 1
                x = x+res
                x = self.relu(x)

        x = x + input
        x = self.relu(x)

        return x

def l2_loss(y_true,y_pred):
    return torch.sum(torch.square((y_true-y_pred)))

def psnr_metric(y_true,y_pred):
    batch_psnr = -10.0 * (1.0/math.log(10)) * torch.log(torch.mean(torch.square(y_pred - y_true),dim=[1,2,3]))
    avg_psnr = torch.mean(batch_psnr)
    return batch_psnr, avg_psnr


if __name__ == "__main__":
    input = torch.ones((2, 1, 128, 128))
    gt = torch.randn((2, 1, 128, 128))
    model = REDNet15()
    # print(model)
    output = model(input)
    loss = l2_loss(y_pred=output, y_true=gt)
    loss.backward()






