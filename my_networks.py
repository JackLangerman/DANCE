import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import torchvision.models
import torchvision.models.resnet as resnet
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models


class scr_net(torch.nn.Module):
    def __init__(self, layers_to_drop=2, num_downsamp_to_drop=2):
        super(scr_net, self).__init__()
        
        model = models.resnet18(pretrained=True)
        layers = list(model.children())[:-layers_to_drop]
        for i in range(num_downsamp_to_drop):
            block = layers[-i-1][0]
            for l in block.children():
                if type(l) is nn.Sequential:
                    for u in l:
                        if type(u) is nn.Conv2d and u.stride == (2,2):
                            u.stride = 1
                elif type(l) is nn.Conv2d and l.stride == (2,2):
                    l.stride = 1
        self.feat_extractor = nn.Sequential(*layers)
        
        # self.regression = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0), 
        #                                 nn.BatchNorm2d(512),
        #                                 nn.LeakyReLU(),
        #                                 nn.Conv2d(512, 3, 1, 1, 0)
        #                                 )
        self.regression = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0), 
                                        nn.BatchNorm2d(512),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(512, 256, 1, 1, 0),
                                        nn.BatchNorm2d(256),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(256, 3, 1, 1, 0)
                                        )
        
    def forward(self, x):
        feat = self.feat_extractor(x)
        
        sc = self.regression(feat)
        return feat, sc

class scr_net_uncertain(torch.nn.Module):
    def __init__(self, layers_to_drop=2, num_downsamp_to_drop=2):
        super(scr_net_uncertain, self).__init__()
        
        model = models.resnet18(pretrained=True)
        layers = list(model.children())[:-layers_to_drop]
        for i in range(num_downsamp_to_drop):
            block = layers[-i-1][0]
            for l in block.children():
                if type(l) is nn.Sequential:
                    for u in l:
                        if type(u) is nn.Conv2d and u.stride == (2,2):
                            u.stride = 1
                elif type(l) is nn.Conv2d and l.stride == (2,2):
                    l.stride = 1
        self.feat_extractor = nn.Sequential(*layers)
        
        self.regression = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0), 
                                        nn.BatchNorm2d(512),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(512, 256, 1, 1, 0),
                                        nn.BatchNorm2d(256),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(256, 3, 1, 1, 0)
                                        )
        self.var = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0), 
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(512, 256, 1, 1, 0),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(256, 1, 1, 1, 0)
                                )
        
    def forward(self, x):
        feat = self.feat_extractor(x)
        
        sc = self.regression(feat)
        var = self.var(feat)

        return feat, sc, var



def CreateDiscriminator(input_channel=512, lr=1e-4, restore_from=None):
    learning_rate_D = lr#1e-4
    discriminator = FCDiscriminator(input_channel)
    optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.9, 0.99))
    optimizer.zero_grad()  
    if restore_from is not None:
        print('loading from pretrained')
        discriminator.load_state_dict(torch.load(restore_from))        
    return discriminator, optimizer


class FCDiscriminator(nn.Module):

    def __init__(self, input_channel, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bce_loss = nn.BCEWithLogitsLoss()


    def forward(self, x, lbl):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        self.loss = self.bce_loss(x, Variable(torch.FloatTensor(x.data.size()).fill_(lbl)).cuda())

        return x

    def adjust_learning_rate(self, args, optimizer, i):
        if args.model == 'DeepLab':
            lr = args.learning_rate_D * ((1 - float(i) / args.num_steps) ** (args.power))
            optimizer.param_groups[0]['lr'] = lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = lr * 10 
        else:
            optimizer.param_groups[0]['lr'] = args.learning_rate_D * (0.1**(int(i/50000)))
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.learning_rate_D * (0.1**(int(i/50000))) * 2              




class FeatureExtractor(torch.nn.Module):
    def __init__(self, 
                 basemodel='resnet18', 
                 pretrained=True, 
                 requires_grad=False, 
                 layers_to_drop=1,
                 num_downsamp_to_drop=0,
                ):
        super(FeatureExtractor, self).__init__()
        model = getattr(torchvision.models, basemodel)(pretrained=pretrained)
        layers = list(model.children())[:-layers_to_drop]
        # unstride downsampling blocks
        for i in range(num_downsamp_to_drop):
            block = layers[-i-1][0]
            for l in block.children():
                if type(l) is nn.Sequential:
                    for u in l:
                        if type(u) is nn.Conv2d and u.stride == (2,2):
                            u.stride = 1
                elif type(l) is nn.Conv2d and l.stride == (2,2):
                    l.stride = 1
        self.layers = nn.ModuleList(layers)#.eval()  
        self.requires_grad_(requires_grad)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(-1).squeeze(-1)
        return x

class MultiHeadModel(nn.Module):
    def __init__(self, feature_extractor, head_1):
        super(MultiHeadModel, self).__init__()
        
        self.f = feature_extractor
        # [*self.heads] = heads
        self.head_1 = head_1
        # self.heads = nn.ModuleList(heads)
        
        
    def forward(self,x):
        x = self.f(x)
        return self.head_1(x)
        # return [h(x) for h in self.heads]

class TaskHead(nn.Module):
    def __init__(self, in_planes, out_planes, planes=None, 
            layers=0, residual=False, global_avarage_pooling=True,
            Block=lambda in_p, p: resnet.Bottleneck(in_p,p//4) ):

        super(TaskHead, self).__init__()

        planes = planes if planes is not None else in_planes
        self.residual = residual
        self.global_avarage_pooling = global_avarage_pooling
        self.out_planes = out_planes

        self.input_transform = nn.Conv2d(in_channels = in_planes, 
                out_channels = planes, 
                kernel_size = 1)

        if planes is not None:
            blocks = [Block(planes, planes) for block in range(layers)]
            if self.residual:
                self.blocks = nn.ModuleList(blocks)
                def progressive_sum(x):
                    for l in self.blocks:
                        x = x + l(x) 
                    return x
                self.layers = progressive_sum
            else:
                self.layers = nn.Sequential(*blocks)
        else:
            self.layers = lambda x: x


        outlist = []
        if self.global_avarage_pooling:
            outlist.append(nn.AdaptiveAvgPool2d(output_size = (1, 1)))
        outlist.append( nn.Conv2d(in_channels = planes, 
                             out_channels = self.out_planes,
                             kernel_size = 1
                            ))
        self.output_transform = nn.Sequential(*outlist)


    def forward(self, x):
        x = self.input_transform(x)
        x = self.layers(x)
        x = self.output_transform(x)
        # x = nn.functional.upsample(x, scale_factor=8, mode='bilinear')
        return x
    

class CoordRegressor2(TaskHead):
    def __init__(self, in_planes, planes=None, 
            layers=0, residual=False,
            Block=lambda in_p, p: resnet.Bottleneck(in_p,p//4) ):

        super(CoordRegressor2, self).__init__(
            in_planes=in_planes, out_planes=3, Block=Block,
            planes=planes, layers=layers, residual=residual, 
            global_avarage_pooling=False, 
            )

def define_SCR():
    extractor = FeatureExtractor(
            basemodel='resnet18',
            pretrained=True,
            requires_grad=True,
            layers_to_drop=2,
            num_downsamp_to_drop=2)
    regressor = CoordRegressor2(extractor(
        torch.zeros(1,3,16,16)).shape[1])
    net = MultiHeadModel(extractor, regressor)
    return net

