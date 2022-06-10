import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl


class UNet_CMs(nn.Module):
    """ Proposed method containing a segmentation network and a confusion matrix network.
    The segmentation network is U-net. The confusion  matrix network is defined in cm_layers
    """
    def __init__(self, in_ch, width, depth, class_no, norm='in', low_rank=False):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # rank: False
        # ===============================================================================
        super(UNet_CMs, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 1
        self.lowrank = low_rank
        #
        self.final_in = class_no
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders_noisy_layers = nn.ModuleList()
        #

        for i in range(self.depth):

            if i == 0:
                #
                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))
                #
            elif i < (self.depth - 1):
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
            else:
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)
        #
        for i in range(self.noisy_labels_no):
            #
            if self.lowrank is False:
                self.decoders_noisy_layers.append(cm_layers(in_channels=width, norm=norm, class_no=self.final_in))
            else:
                self.decoders_noisy_layers.append(low_rank_cm_layers(in_channels=width, norm=norm, class_no=self.final_in, rank=1))

    def forward(self, x):
        #
        y = x
        #
        encoder_features = []
        y_noisy = []
        #
        for i in range(len(self.encoders)):
            #
            y = self.encoders[i](y)
            encoder_features.append(y)
        # print(y.shape)
        for i in range(len(encoder_features)):
            #
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            #
            if y_e.shape[2] != y.shape[2]:
                diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
                diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
                #
                y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            #
            y = self.decoders[-(i+1)](y)
        #
        for i in range(self.noisy_labels_no):
            #
            y_noisy_label = self.decoders_noisy_layers[i](y)
            y_noisy.append(y_noisy_label)
        #
        y = self.conv_last(y)
        #
        if self.training:
            return y, y_noisy
        else:
            return y


class UNet_GlobalCMs(nn.Module):
    """ Baseline with trainable global confusion matrices.
    Each annotator is modelled through a class_no x class_no matrix, fixed for all images.
    """
    def __init__(self, in_ch, width, depth, class_no, input_height, input_width, norm='in'):
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # rank: False
        # input_height: Height of the input image
        # input_width: Width of the input image
        # ===============================================================================
        super(UNet_GlobalCMs, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 4
        self.final_in = class_no
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()

        for i in range(self.depth):

            if i == 0:
                #
                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))
                #
            elif i < (self.depth - 1):
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
            else:
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)

        # Define a list of global confusion matrices:
        # self.decoders_noisy_layers = []
        self.decoders_noisy_layers = nn.ModuleList()
        for i in range(self.noisy_labels_no):
            # self.decoders_noisy_layers.append(global_cm_layers(class_no, input_height, input_width))
            self.decoders_noisy_layers.append(gcm_layers(class_no, input_height, input_width))

    def forward(self, x):
        #
        y = x
        #
        encoder_features = []
        y_noisy = []
        #
        for i in range(len(self.encoders)):
            #
            y = self.encoders[i](y)
            encoder_features.append(y)
        # print(y.shape)
        for i in range(len(encoder_features)):
            #
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            #
            if y_e.shape[2] != y.shape[2]:
                diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
                diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
                #
                y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            #
            y = self.decoders[-(i+1)](y)

        # Return the confusion matrices:
        for i in range(self.noisy_labels_no):
            # Copy the confusion matrix over the batch: (1, c, c, h , w) => (b, c, c, h, w)
            # batch_size = x.size(0)
            # y_noisy_label = self.decoders_noisy_layers[i].repeat(batch_size, 1, 1, 1, 1)
            # y_noisy.append(y_noisy_label.to(device='cuda', dtype=torch.float32))
            y_noisy.append(self.decoders_noisy_layers[i](x))
        #
        y = self.conv_last(y)
        #
        if self.training:
            return y, y_noisy
        else:
            return y


class cm_layers(nn.Module):
    """ This class defines the annotator network, which models the confusion matrix.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)
    """

    def __init__(self, in_channels, norm, class_no):
        super(cm_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_last = nn.Conv2d(in_channels, class_no**2, 1, bias=True)
        self.relu = nn.Softplus()

    def forward(self, x):

        y = self.relu(self.conv_last(self.conv_2(self.conv_1(x))))

        return y


class gcm_layers(nn.Module):
    """ This defines the global confusion matrix layer. It defines a (class_no x class_no) confusion matrix, we then use unsqueeze function to match the
    size with the original pixel-wise confusion matrix layer, this is due to convenience to be compact with the existing loss function and pipeline.
    """

    def __init__(self, class_no, input_height, input_width):
        super(gcm_layers, self).__init__()
        self.class_no = class_no
        self.input_height = input_height
        self.input_width = input_width
        self.global_weights = nn.Parameter(torch.eye(class_no))
        self.relu = nn.Softplus()

    def forward(self, x):

        all_weights = self.global_weights.unsqueeze(0).repeat(x.size(0), 1, 1)
        all_weights = all_weights.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, self.input_height, self.input_width)
        y = self.relu(all_weights)

        return y


class low_rank_cm_layers(nn.Module):
    """ This class defines the low-rank version of the annotator network, which models the confusion matrix at low-rank approximation.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)
    """
    def __init__(self, in_channels, norm, class_no, rank):
        super(low_rank_cm_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_3 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        if rank == 1:
            self.conv_last = nn.Conv2d(in_channels, rank * class_no * 2 + 1, 1, bias=True)
        else:
            self.conv_last = nn.Conv2d(in_channels, rank*class_no*2 + 1, 1, bias=True)
        self.relu = nn.Softplus()

    def forward(self, x):

        y = self.relu(self.conv_last(self.conv_3(self.conv_2(self.conv_1(x)))))

        return y

# =========================
# U-net:
# =========================


class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, step, norm):
        super(conv_block, self).__init__()
        #
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=step, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_1 = nn.PReLU()
        self.activation_2 = nn.PReLU()
        #
        if norm == 'bn':
            self.smooth_1 = nn.BatchNorm2d(out_channels, affine=True)
            self.smooth_2 = nn.BatchNorm2d(out_channels, affine=True)
        elif norm == 'in':
            self.smooth_1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.smooth_2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'ln':
            self.smooth_1 = nn.GroupNorm(out_channels, out_channels, affine=True)
            self.smooth_2 = nn.GroupNorm(out_channels, out_channels, affine=True)
        elif norm == 'gn':
            self.smooth_1 = nn.GroupNorm(out_channels // 8, out_channels, affine=True)
            self.smooth_2 = nn.GroupNorm(out_channels // 8, out_channels, affine=True)

    def forward(self, inputs):
        output = self.activation_1(self.smooth_1(self.conv_1(inputs)))
        output = self.activation_2(self.smooth_2(self.conv_2(output)))
        return output


def double_conv(in_channels, out_channels, step, norm):
    # ===========================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # step: stride
    # ===========================================
    if norm == 'in':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'bn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'ln':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'gn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU()
        )


class UNet(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, norm, dropout=False, apply_last_layer=True):
        """
        Args:
            in_ch:
            width:
            depth:
            class_no:
            norm:
            dropout:
            apply_last_layer:
        """

        # ============================================================================================================
        # This UNet is our own implementation, it is an enhanced version of the original UNet proposed at MICCAI 2015.
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ============================================================================================================
        super(UNet, self).__init__()
        #
        self.apply_last_layer = apply_last_layer
        self.depth = depth
        self.dropout = dropout
        #
        if class_no > 2:
            #
            self.final_in = class_no
        else:
            #
            self.final_in = 1
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        #
        if self.dropout is True:

            self.dropout_layers = nn.ModuleList()

        for i in range(self.depth):

            if self.dropout is True:

                self.dropout_layers.append(nn.Dropout2d(0.4))

            if i == 0:
                #
                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))
                #
            elif i < (self.depth - 1):
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
            else:
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)
        #

    def forward(self, x):
        #
        y = x
        # print(x.shape)
        encoder_features = []
        #
        for i in range(len(self.encoders)):
            #
            y = self.encoders[i](y)
            encoder_features.append(y)
        # print(y.shape)
        for i in range(len(encoder_features)):
            #
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            #
            diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
            diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
            #
            y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            y = self.decoders[-(i+1)](y)
            #
            if self.dropout is True:
                #
                y = self.dropout_layers[i](y)
        #
        if self.apply_last_layer is True:
            y = self.conv_last(y)
        return y




class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)
            # nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            # nn.init.normal_(self.last_layer.bias)

    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])

        del blocks

        # Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)

        if self.apply_last_layer:
            x = self.last_layer(x)

        return x


class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, initializers, padding, pool=True):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding), bias=False))
        layers.append(nn.InstanceNorm2d(output_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding), bias=False))
        layers.append(nn.InstanceNorm2d(output_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, initializers, padding, pool=False)

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)

        if up.shape[3] != bridge.shape[3]:
            #
            diffY = torch.tensor([bridge.size()[2] - up.size()[2]])
            diffX = torch.tensor([bridge.size()[3] - up.size()[3]])
            #
            up = F.pad(up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            # print(up.shape)
            # print(bridge.shape)
            #
        # print(up.shape)
        # print(bridge.shape)

        # assert up.shape[3] == bridge.shape[3]

        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out)

        return out
    
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        # truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        # truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg
