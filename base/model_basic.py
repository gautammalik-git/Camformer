
import torch
from torch import nn
from .model import convolution_output_size, conv_block, lin_block

class CNN(nn.Module):   
    
    """
    Define the neural network structure.
    """
    
    def __init__(self, feature_height, feature_width, batch_size, 
                 print_size=False,
                 input_channels=4,
                 out_channels=[256, 256, 256], 
                 kernels=[(12,1),(4,1),(4,1)], 
                 pool_kernels=[(1,1),(1,1),(1,1)], 
                 paddings=[(0,0), (0,0), (0,0)], 
                 strides=[(1,1), (1,1), (1,1)], 
                 pool_strides=[(1,1), (1,1), (1,1)], 
                 dropouts=[0.2, 0.2, 0.2], 
                 linear_output=[64], 
                 linear_dropouts=[0.0]):
        
        super(CNN, self).__init__()

        if print_size:
            print(locals())
        
        out_size = (feature_height, feature_width) # Set a start size
        
        if print_size:
            print("Sizes: ", end="")
        # Calculate layer output sizes, output if print_size = True
        if paddings == "same":
            paddings = ["same" for x in range(len(out_channels))]
            for n, channels in enumerate(out_channels):
                out_size = convolution_output_size(out_size, pool_kernels[n], (0,0), pool_strides[n])
                if print_size:
                    print(out_size[0],"x",out_size[1],end="\t")
        else:
            for n, channels in enumerate(out_channels):
                out_size = convolution_output_size(out_size, kernels[n], paddings[n], strides[n])
                print(out_size[0],"x",out_size[1],end="\t")
                out_size = convolution_output_size(out_size, pool_kernels[n], (0,0), pool_strides[n])
                if print_size:
                    print(out_size[0],"x",out_size[1],end="\t")

        # Calculate the linear input size.
        output_size = int(out_channels[-1] * out_size[0] * out_size[1]) # Input to the linear layers
        if print_size:
            print(f"\nout_channels[-1]: {out_channels[-1]}")
            print(f"out_sizes: {out_size[0]}, {out_size[1]}")
            print("Linear input size: %s" % output_size)
      
        # Define the convolutional layers
        self.CNN_layers = nn.ModuleList()
        conv_sizes = [x for x in zip([input_channels] + out_channels, out_channels)]
        for n, sizes in enumerate(conv_sizes):
            conv_blocks = conv_block(sizes[0], sizes[1], 
                                     kernel=kernels[n], 
                                     pool_kernel=pool_kernels[n], 
                                     stride=strides[n], 
                                     pool_stride=pool_strides[n], 
                                     padding=paddings[n],
                                     dilation=1, 
                                     dropout=dropouts[n])
            self.CNN_layers.append(nn.Sequential(*conv_blocks))
               
        # Define the fully connected layers
        if len(linear_output) == 0:
            linear_blocks = nn.Sequential(nn.Linear(output_size, 1))
        else:
            linear_sizes = [x for x in zip([output_size] + linear_output, linear_output + [1])]
            linear_blocks = [lin_block(in_f, out_f, dropout=linear_dropouts[n]) for n, (in_f, out_f) in enumerate(linear_sizes[:-1])]
            last_layer = nn.Sequential(nn.Linear(linear_sizes[-1][0], linear_sizes[-1][1]))
            linear_blocks.append(last_layer)
        self.linear_layers = nn.Sequential(*linear_blocks)

    
    def forward(self, x):
        for i in range(0, len(self.CNN_layers)):
            x = self.CNN_layers[i](x)
        
        # FC layers
        x = x.view(x.size(0), -1) # flatten before FC layers
        x = self.linear_layers(x)
        x = torch.flatten(x)
        
        return x
    

