# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


############################ LIBRARIES ######################################
import torch, os, numpy as np

import torch.nn as nn
import pretrainedmodels as ptm

import pretrainedmodels.utils as utils
import torchvision.models as models

import googlenet



"""============================================================="""
def initialize_weights(model):
    """
    Function to initialize network weights.
    NOTE: NOT USED IN MAIN SCRIPT.

    Args:
        model: PyTorch Network
    Returns:
        Nothing!
    """
    for idx,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0,0.01)
            module.bias.data.zero_()



"""=================================================================================================================================="""
### ATTRIBUTE CHANGE HELPER
def rename_attr(model, attr, name):
    """
    Rename attribute in a class. Simply helper function.

    Args:
        model:  General Class for which attributes should be renamed.
        attr:   str, Name of target attribute.
        name:   str, New attribute name.
    """
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


"""=================================================================================================================================="""
### NETWORK SELECTION FUNCTION
def networkselect(opt):
    """
    Selection function for available networks.

    Args:
        opt: argparse.Namespace, contains all training-specific training parameters.
    Returns:
        Network of choice
    """
    if opt.arch == 'googlenet':
        network =  GoogLeNet(opt)
    elif opt.arch == 'resnet50':
        network =  ResNet50(opt)
    else:
        raise Exception('Network {} not available!'.format(opt.arch))
    return network




"""=================================================================================================================================="""
class GoogLeNet(nn.Module):
    """
    Container for GoogLeNet s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt):
        """
        Args:
            opt: argparse.Namespace, contains all training-specific parameters.
        Returns:
            Nothing!
        """
        super(GoogLeNet, self).__init__()

        self.pars       = opt

        self.model = googlenet.googlenet(num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else False)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        rename_attr(self.model, 'fc', 'last_linear')

        self.layer_blocks = nn.ModuleList([self.model.inception3a, self.model.inception3b, self.model.maxpool3,
                                           self.model.inception4a, self.model.inception4b, self.model.inception4c,
                                           self.model.inception4d, self.model.inception4e, self.model.maxpool4,
                                           self.model.inception5a, self.model.inception5b, self.model.avgpool])

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)


    def forward(self, x):
        ### Initial Conv Layers
        x = self.model.conv3(self.model.conv2(self.model.maxpool1(self.model.conv1(x))))
        x = self.model.maxpool2(x)

        ### Inception Blocks
        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = x.view(x.size(0), -1)
        x = self.model.dropout(x)

        mod_x = self.model.last_linear(x)

        #No Normalization is used if N-Pair Loss is the target criterion.
        return mod_x if self.pars.loss=='npair' else torch.nn.functional.normalize(mod_x, dim=-1)



"""============================================================="""
class ResNet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ResNet50, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        mod_x = self.model.last_linear(x)
        #No Normalization is used if N-Pair Loss is the target criterion.
        return mod_x if self.pars.loss=='npair' else torch.nn.functional.normalize(mod_x, dim=-1)
