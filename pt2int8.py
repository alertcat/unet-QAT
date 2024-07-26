import torch
import copy
import re
from torch.quantization import quantize_fx
from model_unet import UnetGenerator
from datasets import CityscapesDataset
import torch.utils.data
from torch import nn

# 自定义模型类以去除 BatchNorm2d
class UnetGeneratorNoBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels):
        super(UnetGeneratorNoBatchNorm, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bridge = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.trans_1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.up_1 = nn.Sequential(
            nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.trans_2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.up_2 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.trans_3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.up_3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.trans_4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.up_4 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        down1 = self.down_1(x)
        pool1 = self.pool_1(down1)
        down2 = self.down_2(pool1)
        pool2 = self.pool_2(down2)
        down3 = self.down_3(pool2)
        pool3 = self.pool_3(down3)
        down4 = self.down_4(pool3)
        pool4 = self.pool_4(down4)
        bridge = self.bridge(pool4)
        trans1 = self.trans_1(bridge)
        up1 = self.up_1(torch.cat([trans1, down4], dim=1))
        trans2 = self.trans_2(up1)
        up2 = self.up_2(torch.cat([trans2, down3], dim=1))
        trans3 = self.trans_3(up2)
        up3 = self.up_3(torch.cat([trans3, down2], dim=1))
        trans4 = self.trans_4(up3)
        up4 = self.up_4(torch.cat([trans4, down1], dim=1))
        out = self.out(up4)
        return out

# 实例化没有 BatchNorm2d 的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "C:\\Users\\26375\\Downloads\\pytorch_unet-master\\unet.pt"

# 定义加载数据集的函数
def load_data(datadir, batch_size):
    dataset = CityscapesDataset(datadir, split='train', mode='fine', augment=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader

# 定义训练函数
def train_model(model, epochs):
    # 这里定义你的训练循环
    datadir = "C:\\Users\\26375\\Downloads\\DeepLabV3Plus-Pytorch-master\\datasets\\data\\cityscapes"
    data_loader = load_data(datadir, batch_size=16)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    model.train()
    for epoch in range(epochs):
        for idx_batch, (imagergb, labelmask, labelrgb) in enumerate(data_loader):
            optimizer.zero_grad()
            x = imagergb.to(device)
            y_ = labelmask.to(device)
            y = model(x)
            y_ = torch.squeeze(y_)
            loss = criterion(y, y_)
            loss.backward()
            optimizer.step()
            if idx_batch % 400 == 0:
                print(f"Epoch [{epoch}/{epochs}], Batch [{idx_batch}], Loss: {loss.item()}")

if __name__ == '__main__':
    # 加载提前训练好的模型参数
    model = UnetGeneratorNoBatchNorm(3, 4, 64).to(device)  # 3输入通道，4输出通道，64是基础通道数
    checkpoints = torch.load(model_path, map_location=device)

    # 去除state_dict中的module前缀
    new_state_dict = {}
    for key, value in checkpoints.items():
        new_key = re.sub(r'^module\.', '', key)
        new_state_dict[new_key] = value

    # 移除不匹配的参数
    model_dict = model.state_dict()
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

    # 更新现有的model state_dict
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    # 接下来直接进行QAT准备和训练
    print('Begin QAT...')
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.train()  # Set model mode to train

    # Get default qconfig
    qconfig_dict = {"": torch.quantization.get_default_qconfig('qnnpack')}

    # 创建example_inputs
    example_inputs = torch.randn(1, 3, 128, 256).to(device)  # 示例输入，假设输入图像大小为128x256

    # Prepare model
    model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_dict, example_inputs=example_inputs)

    model_prepared.to(device)  # 模型拷贝至用于训练的device
    train_model(model_prepared, 10)  # 使用与Float32模型同样的训练函数，对prepared模型继续训练若干轮，这里为了方便，我只设置了10轮
    torch.save(model_prepared.state_dict(), 'model_prepared.pth')  # 保存prepared模型

    # Convert model to int8 (在CPU上进行)
    print('Converting model to int8...')
    model_prepared.to('cpu')
    model_quantized = quantize_fx.convert_fx(model_prepared)  # 将prepared模型转换成真正的int8定点模型
    print('Convert done.')
    torch.save(model_quantized.state_dict(), 'model_int8.pth')  # 保存定点模型的state_dict
