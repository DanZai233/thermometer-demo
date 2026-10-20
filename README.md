# 测温证据审核 Demo

本地网页：上传维修前后照片，运行整图AI检测、局部编辑定位、照片查重与工单一致性审核。检测运行失败会明确报错，未运行时没有真实性结论。

## 技术栈与目录

- 应用：Python 3.11、Streamlit 单进程本地网页，无独立前端或数据库服务。
- 图片处理：Pillow 负责读取、EXIF 方向处理与裁剪；ImageHash 做 pHash 相似筛查；SHA-256 做精确查重。
- 本地推理：PyTorch、torchvision、timm、safetensors；Community Forensics 检测整图 AI 迹象；CAT-Net v2 定位编辑响应，jpeglib 读取 JPEG DCT，yacs/PyYAML 处理模型配置。支持 CPU、Apple MPS、CUDA。
- 可选 API：Requests 调用 DeepSeek / OpenAI 兼容视觉接口，仅在用户勾选后发送照片，用于温度、水印信息提取。
- 存储：本地图片与 JSON 报告，无数据库。界面依赖见 `requirements.txt`，其中包含 `requirements-models.txt` 的推理依赖。
- 入口：`app.py`；业务和 API：`core.py`；本地模型：`local_models.py`；第三方模型代码：`vendor/catnet/`；权重来源与哈希：`models/provenance.json`。

## 真实模型

- Community Forensics 384：官方MIT权重，基于ViT-small，输出未校准AI响应分数。输入遵循作者Resize(440)+CenterCrop(384)。中心裁剪可能漏掉边缘的仪表，整图低分不能排除局部AI编辑。
- CAT-Net v2：官方Google Drive完整checkpoint，严格加载网络权重。CPU、Apple MPS、CUDA均有推理入口。原始JPEG取出真实DCT系数；非JPEG按作者处理方式转JPEG，显示相应限制。
- CAT-Net以原分辨率512像素切块、384步长和重叠平均执行，保持JPEG的8×8网格；这是本应用工程适配，不是官方整图评测协议。响应图使用固定0至1色标，不做自适应拉伸。红框、水印可引发编辑响应，不自动等于温度数字造假。
- 可在屏幕放大区域调整选框，检查该屏幕的编辑响应。样例初始选框由人工观察设置，可修改。
- 阈值（AI 0.2/0.8，局部响应0.5、区域占比等）仅为演示规则，没有经过温度计专项校准，不显示成“造假概率”。

Fleet未接入。DeepSeek仅提取读数和水印，不参与本地取证判断。

## 启动

Python 3.11+，uv可从官方包管理器安装。初次需安装PyTorch等依赖。

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

打开 http://127.0.0.1:8501 。关闭服务在终端按 Ctrl-C。

本次已下载的模型位于models/。新机器仅安装 Python 依赖不会自动取得权重；需按 `models/provenance.json` 的来源准备模型文件，并在页面确认权重就绪。先加载案例，再点击“检测 AI 生成 / 局部 P 图（本地运行）”；本地检测不需要Key，不发送照片。首次推理需要加载权重，耗时取决于图片和设备；同一照片复用缓存。

## DeepSeek

默认供应商https://api.deepseek.com，模型deepseek-flash。侧栏填写Key或设置DEEPSEEK_API_KEY，并勾选允许发送照片后执行“提取照片信息”。请求关闭思考模式，使用JSON输出与high图像细节。没有Key或未勾选时不发送请求。
官方视觉文档：https://api-docs.deepseek.com/guides/vision
自定义OpenAI兼容服务可配置VISION_BASE_URL、VISION_MODEL、VISION_API_KEY。API Key不写入报告。

## 数据及报告

samples/来自用户提供的真实业务附件，含个人信息，仅用于本地演示，请勿公开。样例读数、水印和屏幕框为人工观察，不是自动检测；照片阶段未确认。
原件存于data/，SHA-256用于文件完整性和精确查重。报告在data/reports/，检测缓存也在data/。上传后哈希不能证明上传前未修改。pHash仅筛相似图片，尚不能可靠处理裁剪复用。

## 模型来源和许可

CAT-Net：https://github.com/mjkwon2021/CAT-Net
作者README链接的Google Drive：trained weight/CAT_full_v2.pth.tar。
代码及第三方声明见vendor/catnet/NOTICE.md与LICENSE-HRNet.txt。作者README声明代码Apache-2.0、权重CC-BY-4.0，第三方许可继续适用。
Community Forensics：https://github.com/JeongsooP/Community-Forensics
权重：https://huggingface.co/OwensLab/commfor-model-384，MIT。
下载来源、版本和文件SHA256记录在models/provenance.json。

## 外部适配器

可设置FORENSIC_ADAPTER为本地可执行文件路径以覆盖CAT-Net。调用参数为“适配器 原始图片路径 输出目录”，不经shell。适配器须在输出目录写result.json，包含model和输出目录内的heatmap绝对路径。只有真实推理成功才能写入结果。

## 测试与限制

```bash
.venv/bin/python -m unittest test_core.py
```

当前模型结果不是司法鉴定。静态照片不能证明测量稳定、探针位置、仪表校准和当时室温。业务验收温度阈值尚未配置；不自动判断供热达标。应使用来源明确的真实/传统修改/AI修改/无害水印样本做独立专项测试，再选择上线阈值。

## 本次实测限制

已在Apple MPS运行全部5张案例、2个模型。额外做了同一案例图的JPEG重编码对照和人为数字粘贴版本：屏幕强响应占比均约99.23%，未明显区分。因此该实现完成了真实模型接入，但尚未验证温度数字篡改检测能力，不可用于自动判罚或验收。结果见controlled_validation.json。热力站显示屏整图AI响应约0.974，属于特殊场景可能误报，应单独审核。

## 版本管理状态

项目远端为私有仓库 `DanZai233/thermometer-demo`。`samples/` 含真实业务附件；对外发布前必须排除这些素材及含业务信息的截图、报告，同时排除 `data/`、模型大文件和密钥。当前文档覆盖本机运行，但还不是在全新机器验证过的完整自动部署流程。

仓库仅包含代码、依赖和文档。真实案例、截图、生成报告、样例媒体与模型大文件保留在本机，不随 Git 分发；克隆后需另行准备权重和有权使用的测试素材。
