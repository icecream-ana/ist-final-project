# 信息技术安全期末大作业

## 组员信息
温可童（23336244）、林珺怡（22306035）、何翌宁（23336085）

## 分工
- 何翌宁：负责搜索 2–3 篇具有开源代码或较高复现可行性且具备可比性的论文；撰写报告大纲；负责报告中“背景与动机”“相关工作”“两篇论文复现方案详述”“两篇论文复现后结果的综合对比分析”以及“总结”部分的撰写，并对整篇实验报告进行整体润色和格式统一。

- 林珺怡：负责复现论文 Tree-Ring Watermarks，按照既定实验设计运行代码，获取并整理代运行效果截图；撰写报告中“实验验证与结果分析”和“代码运行截图”中论文一的复现方案、实验结果等相关章节，以及整体论文润色。

- 温可童：负责复现论文 The Stable Signature，按照既定实验设计运行代码，获取并整理代码运行效果截图；撰写报告中“实验验证与结果分析”和“代码运行截图”中论文二的复现方案、实验结果等相关章节，以及整体论文润色。

## 文件结构
**整体文件结构：**
```
├── README.md                         # 项目说明
├── code                              # 代码
│   ├── stable_signature.zip          # The Stable Signature 代码
│   └── tree_ring_watermarks.zip      # Tree-Ring Watermarks 代码
├── 实验结果                           # 官方 LDM decoder 微调脚本
│   ├── stable_signature
│       ├── img_metrics.csv           # 评估图像质量完整结果
│       ├── log_stats.csv             # 评估水印可检测性完整结果
│       ├── 生成图片.mp4               # 生成图片过程的视频
│       ├── 微调LDM解码器.mp4          # 微调LDM解码器过程的视频
│       ├── 实验二：评估水印可检测性.jpg# 评估水印可检测性的终端截图
│       └── 实验二：评估图像质量.jpg    # 评估图像质量的终端截图
│   └── tree_ring_watermarks
│       ├── log                       # 三个实验日志及结果
│       ├── 单个脚本运行.mp4           # 单个py程序运行过程的视频
│       ├── run_table1批量脚本运行.mp4 # 多个py程序批量运行脚本的视频
└── 实验报告.pdf
```

**代码stable_signature的结构：**
```
stable_signature/
├── README.md                         # 项目说明与官方复现步骤
├── requirements.txt                  # Python 依赖列表
├── finetune_ldm_decoder.py           # 官方 LDM decoder 微调脚本
├── finetune_ldm_decoder_lowvram.py   # 本实验新增的低显存 decoder 微调脚本
├── generate_sd15_watermarked.py      # 本实验新增的带/不带水印图像生成脚本
├── run_evals.py                      # 水印鲁棒性与图像质量评估脚本
├── decoding.ipynb                    # 官方水印解码示例 notebook
├── utils.py                          # 通用工具函数
├── utils_img.py                      # 图像处理、攻击变换和质量指标函数
├── utils_model.py                    # 模型构建与 checkpoint 加载函数
├── download_coco_subset.py           # COCO 子集下载辅助脚本
│
├── models/                           # 官方水印提取器模型权重
│   ├── dec_48b_whit.torchscript.pt   # 主要使用的 48-bit 水印提取器
│   └── other_dec_48b_whit.torchscript.pt # 另一版鲁棒性提取器
│
├── data/                             # 数据集与 COCO 标注文件
│   ├── train/                        # decoder 微调用训练图像
│   ├── val/                          # decoder 微调用验证图像
│   ├── annotations/                  # 解压后的 COCO 标注文件
│   └── annotations_trainval2017.zip  # COCO 2017 标注压缩包
│
├── sd/                               # Stable Diffusion 模型文件
│   └── stable-diffusion-v1-5/
│       ├── v1-inference.yaml         # SD v1.5 模型结构配置
│       └── v1-5-pruned-emaonly.ckpt  # SD v1.5 checkpoint 权重
│
├── hf_models/                        # 本地缓存的 Hugging Face 模型
│   └── clip-vit-large-patch14/
│       ├── config.json               # CLIP 模型配置
│       ├── pytorch_model.bin         # CLIP 模型权重
│       ├── tokenizer_config.json     # tokenizer 配置
│       ├── vocab.json                # tokenizer 词表
│       └── merges.txt                # BPE 合并规则
│
├── openai/                           # 供 diffusers 离线查找的 CLIP 文件
│   └── clip-vit-large-patch14/
│       ├── config.json               # CLIP 配置文件
│       ├── tokenizer_config.json     # tokenizer 配置
│       ├── special_tokens_map.json   # 特殊 token 映射
│       ├── vocab.json                # tokenizer 词表
│       └── merges.txt                # BPE 合并规则
│
├── hidden/                           # HiDDeN 水印模型训练代码
│   ├── main.py                       # HiDDeN encoder/decoder 训练入口
│   ├── models.py                     # HiDDeN 水印模型结构
│   ├── data_augmentation.py          # 水印训练中的数据增强
│   ├── attenuations.py               # 水印信号衰减相关模块
│   ├── ckpts/                        # HiDDeN 预训练 checkpoint
│   ├── imgs/                         # 示例图像
│   └── notebooks/                    # HiDDeN 示例 notebook
│
└── src/                              # 核心源码目录
    ├── ldm/                          # Latent Diffusion / Stable Diffusion 代码
    │   ├── models/                   # AutoencoderKL 和 diffusion 模型
    │   ├── modules/                  # attention、encoder、diffusion 模块
    │   └── data/                     # LDM 数据加载相关代码
    │
    ├── taming/                       # VQGAN / Taming Transformers 依赖代码
    │   ├── models/                   # VQGAN 相关模型
    │   ├── modules/                  # VQGAN 网络模块与损失
    │   └── data/                     # 数据集处理代码
    │
    └── loss/                         # 感知损失与图像质量损失
        ├── loss_provider.py          # 感知损失统一构建入口
        ├── watson_vgg.py             # Watson-VGG 感知损失
        ├── watson_fft.py             # Watson-FFT 感知损失
        ├── watson.py                 # Watson-DCT 感知损失
        ├── ssim.py                   # SSIM 损失
        └── losses/                   # PerceptualSimilarity 预训练权重
```

**代码tree_ring_watermarks的结构：**
```
tree_ring_watermark/
├── guided_diffusion/                 # 引导扩散模型库 (OpenAI)
│   ├── __init__.py
│   ├── dist_util.py
│   ├── fp16_util.py
│   ├── gaussian_diffusion.py
│   ├── image_datasets.py
│   ├── logger.py
│   ├── losses.py
│   ├── nn.py
│   ├── resample.py
│   ├── respace.py
│   ├── script_util.py
│   ├── train_util.py
│   └── unet.py
│
├── open_clip/                        # OpenCLIP 模型库
│   ├── __init__.py
│   ├── coca_model.py
│   ├── constants.py
│   ├── factory.py
│   ├── generation_utils.py
│   ├── hf_configs.py / hf_model.py
│   ├── loss.py
│   ├── model.py
│   ├── modified_resnet.py
│   ├── openai.py / pretrained.py / timm_model.py
│   ├── tokenizer.py
│   ├── transform.py / transformer.py
│   ├── utils.py / version.py
│   └── model_configs/               # 模型配置文件 (~50个JSON)
│
├── pytorch_fid/                      # FID 评分计算库
│   ├── __init__.py / __main__.py
│   ├── fid_score.py
│   └── inception.py
│
├── src/tree_ring_watermark/          # 核心: 树环水印实现
│   ├── __init__.py
│   ├── _check.py                     # 水印验证
│   ├── _detect.py                    # 水印检测
│   ├── _get_noise.py                 # 噪声生成
│   ├── target_api.py                 # 目标API接口
│   ├── utils.py                      # 工具函数
│   └── dependency_versions_table.py
│
├── scripts/                          # Shell 脚本
│   ├── tree_ring.sh
│   ├── tree_rand.sh
│   ├── tree_zeros.sh
│   └── tree_ring_imagenet.sh
│
├── openai_config/                    # OpenAI 配置文件
│   └── 256x256_diffusion.json
│
├── log/                              # 运行日志
├── fid_outputs/                      # FID 输出结果
├── wandb/                            # W&B 训练日志
│
├── run_tree_ring_watermark.py        # 主入口: 编码+解码水印
├── run_tree_ring_watermark_fid.py    # 主入口: 带 FID 评估的水印实验
├── run_tree_ring_watermark_imagenet.py        # 主入口: ImageNet 数据集版本
├── run_tree_ring_watermark_imagenet_fid.py    # 主入口: ImageNet + FID 版本
├── inverse_stable_diffusion.py       # 逆向稳定扩散
├── modified_stable_diffusion.py      # 修改版稳定扩散
├── optim_utils.py                    # 优化工具
├── io_utils.py                       # 输入输出工具
├── setup.py                          # 安装脚本
├── run_table1.bat / .sh              # 实验1批量运行脚本
├── run_table2.bat / .sh              # 实验2批量运行脚本
├── run_table3.bat / .sh              # 实验3批量运行脚本
├── requirements.txt
├── README.md
└── 256x256_diffusion.json            # 扩散模型配置
```
