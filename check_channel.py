import torch
import re
from model_unet import UnetGenerator

# 假设模型保存的路径
model_path = "C:\\Users\\26375\\Downloads\\pytorch_unet-master\\unet.pt"

# 实例化模型
model = UnetGenerator(3, 4, 64)  # 假设已知输入通道为3，输出通道为4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型参数
checkpoints = torch.load(model_path, map_location=device)

# 去除state_dict中的module前缀
new_state_dict = {}
for key, value in checkpoints.items():
    new_key = re.sub(r'^module\.', '', key)
    new_state_dict[new_key] = value

# 加载新state_dict到模型中
model.load_state_dict(new_state_dict)

# 打印模型结构
print(model)

# 遍历所有层并打印卷积层的输入和输出通道数
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        print(f"Layer {name}: input channels: {layer.in_channels}, output channels: {layer.out_channels}")
