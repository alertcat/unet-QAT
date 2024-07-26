import torch

# 加载 .pkl 文件
model = torch.load("C:\\Users\\26375\\Downloads\\pytorch_unet-master\\unet.pkl", map_location=torch.device('cuda'))

# 保存为 .pt 文件
torch.save(model.state_dict(), "C:\\Users\\26375\\Downloads\\pytorch_unet-master\\unet.pt")

print("转换完成！")