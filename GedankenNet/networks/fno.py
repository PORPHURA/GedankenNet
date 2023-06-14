import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # out_ft = self.ca(out_ft.abs()) * out_ft
        # out_ft = self.sa(out_ft.abs()) * out_ft

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes, width, in_dim, out_dim=2):
        super(FNO2d, self).__init__()
        
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.conv_begin_0 = nn.Conv2d(in_dim+2, self.width, 1)  # GoPro dataset: RGB 3 channels
        self.conv_begin_1 = nn.Conv2d(self.width, self.width, 1)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv_end1 = nn.Conv2d(self.width, self.width, 1)
        self.conv_end2 = nn.Conv2d(self.width, 2, 1)
        self.prelu_begin = nn.PReLU(self.width)
        self.prelu_end = nn.PReLU(self.width)
        self.lsc = nn.Conv2d(self.width, self.width, 1)

        self.scales_per_block = [1,1,2,2,2,4,4,4]
        self.share_block = [True,True,True,True,True,False,False,False]
        self.num_per_block = [2,2,2,2,2,2,2,2]
        assert len(self.scales_per_block) == len(self.share_block) and len(self.scales_per_block) == len(self.num_per_block)

        self.SConv2d_list = []
        self.w_list = []
        self.prelu_list = []
        self.ssc_list = []
        for i in range(len(self.scales_per_block)):
            print("building scales", i)
            if self.share_block[i]:
                print("\tshared params", end=' ')
                print(self.scales_per_block[i])
                self.SConv2d_list.append(SpectralConv2d_fast(self.width, self.width, self.modes//self.scales_per_block[i], self.modes//self.scales_per_block[i]))
                self.w_list.append(nn.Conv2d(self.width, self.width, 1))
                self.prelu_list.append(nn.PReLU(self.width))
            else:
                print("\tnot shared params", end=' ')
                for _ in range(self.num_per_block[i]):
                    print(self.scales_per_block[i], end=' ')
                    self.SConv2d_list.append(SpectralConv2d_fast(self.width, self.width, self.modes//self.scales_per_block[i], self.modes//self.scales_per_block[i]))
                    self.w_list.append(nn.Conv2d(self.width, self.width, 1))
                    self.prelu_list.append(nn.PReLU(self.width))
                print()
            self.ssc_list.append(nn.Conv2d(self.width, self.width, 1))
        self.SConv2d_list = nn.ModuleList(self.SConv2d_list)
        self.w_list = nn.ModuleList(self.w_list)
        self.prelu_list = nn.ModuleList(self.prelu_list)
        self.ssc_list = nn.ModuleList(self.ssc_list)
        

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        # x = x.permute(0, 3, 1, 2)
        x = self.conv_begin_0(x)
        x = self.prelu_begin(x)
        x = self.conv_begin_1(x)
        x_0 = x
        x_s = x_0
        x_t = x_0
        

        pointer = 0
        for i in range(len(self.scales_per_block)):
            # print(i)
            if self.share_block[i]:
                for _ in range(self.num_per_block[i]):
                    # print("\t",pointer)
                    x = self.SConv2d_list[pointer](x) + self.w_list[pointer](x)
                    x = self.prelu_list[pointer](x)
                    x = x + x_t
                    x_t = x
                pointer += 1
            else:
                for _ in range(self.num_per_block[i]):
                    # print("\t",pointer)
                    x = self.SConv2d_list[pointer](x) + self.w_list[pointer](x)
                    x = self.prelu_list[pointer](x)
                    x = x + x_t
                    x_t = x
                    pointer += 1
            x = self.ssc_list[i](x)
            x = x + x_s
            x_s = x
            x_t = x_s


        x = self.lsc(x)
        x = x + x_0

        x = self.conv_end1(x)
        x = self.prelu_end(x)
        x = self.conv_end2(x)
        # x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        # return [N, C, H, W] grids
        n, c, size_x, size_y = shape
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([n, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([n, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
