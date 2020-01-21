import torchvision.models as models
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch

class Model_ImageNet(nn.Module):
    def __init__(self, num_classes):
        super(Model_ImageNet, self).__init__()
        base = models.__dict__['resnet34'](pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, input):
        feat = self.base(input).squeeze()
        output = self.fc1(feat)
        return output


class Model_Random(nn.Module):
    def __init__(self, num_classes):
        super(Model_Random, self).__init__()
        base = models.__dict__['resnet34']()
        self.base = nn.Sequential(*list(base.children())[:-1])
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, input):
        feat = self.base(input).squeeze()
        output = self.fc1(feat)
        return output

# class Model_SelfSupervised(nn.Module):
#     def __init__(self, num_classes, pretrain_task):
#         super(Model_SelfSupervised, self).__init__()
#
#         #I should create a python file that it can just pull all of the information it needs to create the model for each specific pretraining task
#
#         warmup_dict =torch.load(pretrain_path)
#         pretrainedm = Model_Random(num_classes=num_classes)
#         pretrainedm.load_state_dict(warmup_dict[0])
#         pretrainedm.opt = optim.Adam(pretrainedm.parameters(), lr=args.lr, weight_decay=args.wd)
#         pretrainedm.opt.load_state_dict(warmup_dict[1])
#
#         self.base = nn.Sequential(*list(pretrainedm.base.children())[:])
#         self.fc1 = nn.Linear(512, 2)
#
#
#         print(num_classes)
#
#     def forward(self, input):
#         feat = self.base(input).squeeze()
#         output = self.fc1(feat)
#         return output
#

