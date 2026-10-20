CAT-Net官方网络定义及配置，来源：https://github.com/mjkwon2021/CAT-Net
Commit: 331b8059c3f55efec1d9075de79dd153413f2061
文件network_CAT.py与CAT_full.yaml未修改，保留原文件版权注释及HRNet许可。
官方README于2026-08-05声明CAT-Net代码Apache-2.0、权重CC-BY-4.0；原始HRNet代码仍遵守MIT及相关第三方许可。
本应用新编写的CPU/MPS推理、DCT预处理和切块聚合代码在local_models.py，不是官方评测脚本。
CAT-Net权重从作者README链接的Google Drive trained weight/CAT_full_v2.pth.tar下载。
Community Forensics官方权重：https://huggingface.co/OwensLab/commfor-model-384，模型卡MIT。
其模型架构和输入预处理遵循官方models.py、dataloader.py定义，采用timm构建并严格加载全部权重。
