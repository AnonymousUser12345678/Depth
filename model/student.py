import torch
import torch.nn as nn
import torchvision.models as models


class Student(nn.Module):
    def __init__(self, args):
        super(Student, self).__init__()
        
        if args.student_arch == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet34(pretrained=True)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.loss = args.loss
        if args.global_info:
            self.global_layer = nn.Linear(256*4, args.student_input_size*args.student_input_size)
        self.global_info = args.global_info
        
        # 2 * (16 - 1) + 4 - 2 * 1 = 32
        # 2 * (32 - 1) + 4 - 2 * 1 = 64
        # s_o = stride * (s_i - 1) + kernel_size - 2 * padding
        if args.student_input_size == 64:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            )

    def forward(self, x, x2=None):
        x = self.encoder(x)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.decoder(x)
        
        if self.global_info:
            x2 = x2.view(x2.size(0), -1)
            x2 = self.global_layer(x2)
            x2 = x2.view(x.size(0), 1, x.shape[-1], x.shape[-1])
            
            x = x * x2
        return x