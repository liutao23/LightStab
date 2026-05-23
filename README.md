# No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors

<p align="center">
  <b>CVPR 2026</b><br>
  Official implementation for <br>
  <b>No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors</b>
</p>

<p align="center">
  <a href="https://cvpr.thecvf.com/virtual/2026/poster/39183"><img src="https://img.shields.io/badge/CVPR-2026-4b44ce.svg" alt="CVPR 2026"></a>
  <a href="https://arxiv.org/abs/2602.23141"><img src="https://img.shields.io/badge/arXiv-2602.23141-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.48550/arXiv.2602.23141"><img src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2602.23141-blue.svg" alt="DOI"></a>
  <a href="https://www.youtube.com/watch?v=SBrtgR3HAJo"><img src="https://img.shields.io/badge/Video-YouTube-red.svg" alt="YouTube video"></a>
</p>

<p align="center">
  <a href="https://cvpr.thecvf.com/virtual/2026/poster/39183">CVPR Page</a> •
  <a href="https://arxiv.org/abs/2602.23141">arXiv</a> •
  <a href="https://arxiv.org/pdf/2602.23141">Paper PDF</a> •
  <a href="#poster-and-video">Poster & Video</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#citation">Citation</a> •
  <a href="#中文">中文</a>
</p>

---

## Table of Contents

- [English](#english)
  - [News](#news)
  - [Overview](#overview)
  - [Poster and Video](#poster-and-video)
  - [Highlights](#highlights)
  - [Method](#method)
  - [Project Status](#project-status)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Project Structure](#project-structure)
  - [TODO](#todo)
  - [Important Note](#important-note)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)
  - [FAQ](#faq)
- [中文](#中文)
  - [更新日志](#更新日志)
  - [项目简介](#项目简介)
  - [海报与视频](#海报与视频)
  - [项目特点](#项目特点)
  - [方法说明](#方法说明)
  - [当前状态](#当前状态)
  - [安装方法](#安装方法)
  - [快速开始](#快速开始)
  - [项目结构](#项目结构)
  - [待办事项](#待办事项)
  - [重要说明](#重要说明)
  - [致谢](#致谢)
  - [引用](#引用)
  - [常见问题](#常见问题)

---

# English

## News

- **[2026-05]** CVPR 2026 virtual poster page and presentation video are available.
- **[2026-02]** Paper released on arXiv.
- **[2026-03]** Repository initialized.
- **[2026-03]** Online stabilization inference code released.
- **[Coming Soon]** Code cleanup, full dataset release, and training scripts.

---

## Overview

This repository contains the official implementation of our CVPR 2026 paper:

**No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors**

We propose an **unsupervised online video stabilization** framework that does not rely on paired stable / unstable training labels and does not use future-frame look-ahead during inference. The method instantiates a classical stabilization pipeline with practical priors and a multithreaded buffering mechanism, targeting causal online deployment and resource-constrained hardware.

This work is heavily inspired by [DUTCode](https://github.com/Annbless/DUTCode.git). We sincerely thank the original authors for their generous open-source contribution and pay tribute to their excellent work. Up to now, we still believe that **DUTCode remains one of the best video stabilization methods**.

---

## Poster and Video

- [CVPR 2026 virtual poster page](https://cvpr.thecvf.com/virtual/2026/poster/39183)
- [arXiv page](https://arxiv.org/abs/2602.23141)
- [Paper PDF](https://arxiv.org/pdf/2602.23141)
- [YouTube presentation](https://www.youtube.com/watch?v=SBrtgR3HAJo)

<!--
If the CVPR-hosted poster image does not render before the CVPR virtual site finishes updating,
replace the src below with a local file such as assets/poster.png.
GitHub supports relative image paths in Markdown files.
-->

<p align="center">
  <a href="https://cvpr.thecvf.com/virtual/2026/poster/39183">
    <img src="https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202026/39183.png" alt="CVPR 2026 Poster" width="95%">
  </a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=SBrtgR3HAJo">
    <img src="https://img.youtube.com/vi/SBrtgR3HAJo/maxresdefault.jpg" alt="Watch the CVPR 2026 presentation video" width="70%">
  </a>
</p>

---

## Highlights

- **Unsupervised** video stabilization
- **Online** inference without future-frame look-ahead
- Built upon strong **classical priors**
- Lightweight and practical adaptation for real-time / online scenarios
- Multithreaded buffering for efficient causal processing
- UAV-Test benchmark for multimodal UAV aerial video stabilization

---

## Method

Our method focuses on **unsupervised online video stabilization without labels and without future-frame look-ahead**.

Unlike offline stabilization methods that rely on future information, this work targets a more practical online setting, where each frame is processed causally. The overall framework follows the classical stabilization pipeline and incorporates practical priors for motion estimation, motion propagation, and motion compensation.

The design addresses three practical challenges that often limit end-to-end learning-based stabilizers:

1. limited availability of paired stable / unstable video data;
2. reduced controllability in fully learned pipelines;
3. inefficient deployment on resource-constrained hardware.

More technical details, visual results, and ablation studies will be added as the repository is cleaned and updated.

---

## Project Status

- Paper: **accepted to CVPR 2026**
- arXiv preprint: **available**
- CVPR virtual poster page: **available**
- YouTube presentation: **available**
- Inference / online demo: **available**
- Test dataset (**UAV-Test**): **available**
- Training scripts: **not released yet**
- Full training dataset: **coming soon**

---

## Installation

### 1. Download assets

Please download the `LightStab_assets` package and copy **all files** directly into the project root directory.

These files include pretrained weights and required runtime assets.

[Download LightStab_assets](https://drive.usercontent.google.com/download?id=1pHD3BR2KXKHjksKTx5z50HAE-2GNOO17&export=download&authuser=0&confirm=t&uuid=cd5409a4-0e8e-49f4-8189-23ef4f6ea6c1&at=AGN2oQ07Ev9BOwUa2gHbhyxK3fr3:1774845474015)

### 2. Create and activate the environment

```bash
conda env create -n lightstab -f environment.yaml
conda activate lightstab
```

---

## Quick Start

Run the online stabilization script:

```bash
cd scripts
python onlinestab.py
```

---

## Project Structure

A typical project layout is as follows:

```text
.
├── scripts/
│   └── onlinestab.py
├── environment.yaml
├── README.md
└── ...
```

The exact structure may evolve as the repository is further cleaned and updated.

---

## TODO

- [x] Release initial inference code
- [x] Provide runtime assets / pretrained weights
- [x] Release the test dataset (**UAV-Test**)
- [x] Add CVPR virtual poster page link
- [x] Add YouTube presentation link
- [ ] Release a cleaned repository version
- [ ] Release the full training dataset
- [ ] Release training scripts
- [ ] Add qualitative visualization results
- [ ] Add more detailed documentation
- [ ] Add updates for the journal-extension version

---

## Important Note

We are currently preparing an extended journal version of this work. For this reason, the **training scripts are not publicly available at this stage**.

We expect to release the **full training code and dataset** after the journal extension and submission process is completed, which is currently planned for **around late August**.

Before that, **please do not email us to request the training scripts**, as we will not be able to reply to such requests individually.

Thank you for your understanding and support.

---

## Acknowledgements

We gratefully acknowledge the following excellent repositories and projects that inspired this work:

- [Grundmann et al.](https://github.com/ishank-juneja/L1-optimal-paths-Stabilization.git)
- [Bundle](https://github.com/SuTanTank/BundledCameraPathVideoStabilization.git)
- [DIFRINT](https://github.com/jinsc37/DIFRINT.git)
- [Yu and Ramamoorthi](https://jiyang.fun/projects.html)
- [PWStableNet](https://github.com/mindazhao/PWStableNet.git)
- [DUT](https://github.com/Annbless/DUTCode.git)
- [Deep3D](https://github.com/yaochih/Deep3D-Stabilizer-release.git)
- [FuSta](https://github.com/alex04072000/FuSta.git)
- [RStab](https://github.com/pzzz-cv/RStab.git)
- [MetaStab](https://github.com/MKashifAli/MetaVideoStab.git)
- [GaVS](https://github.com/huawei-bayerlab/GaVS.git)
- [MeshFlow](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization.git)
- [StabNet](https://github.com/cxjyxxme/deep-online-video-stabilization-deploy.git)
- [NNDVS](https://github.com/liuzhen03/NNDVS.git)
- [Liu et al.](https://github.com/liutao23/Realtime_Video_Stabilization.git)

Special thanks to the authors of **DUTCode** for their inspiring open-source contribution.

---

## Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{liu2026nolabelsnolookahead,
  title={No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors},
  author={Liu, Tao and Wan, Gang and Ren, Kan and Wen, Shibo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

@article{xu2022dut,
  title={Dut: Learning video stabilization by simply watching unstable videos},
  author={Xu, Yufei and Zhang, Jing and Maybank, Stephen J. and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={4306--4320},
  year={2022},
  publisher={IEEE}
}
```

The arXiv version is available at [arXiv:2602.23141](https://arxiv.org/abs/2602.23141).

---

## FAQ

### Q1: Are the training scripts available?

Not yet. We plan to release them after the journal extension and submission process is completed.

### Q2: Is the dataset available now?

At present, only the paper's test dataset **UAV-Test** is available:

[Download UAV-Test](https://drive.usercontent.google.com/download?id=1eHJ4Z8uqPKheDDzea4nQWGwwq5aauBvh&export=download&authuser=0&confirm=t&uuid=f05ae258-5403-4dec-add8-27ce19a96fab&at=AGN2oQ1ce8ktgFS0Qhf1KN_8P4dF:1775119978751)

The training data does not consist of raw video frames, but rather extracted keypoints and optical flow. We still need time to organize these materials and release them together with the full dataset.

### Q3: Can I request the training scripts by email?

Please do not email us to request the training scripts at this stage. Such requests will not be replied to individually.

---

# 中文

## 更新日志

- **[2026-05]** 已添加 CVPR 2026 虚拟海报页面和报告视频链接。
- **[2026-02]** 论文已发布至 arXiv。
- **[2026-03]** 初始化仓库。
- **[2026-03]** 发布在线视频稳定推理代码。
- **[即将发布]** 代码整理、完整数据集和训练脚本。

---

## 项目简介

本仓库是 CVPR 2026 论文的官方实现：

**No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors**

本文提出一种**无监督在线视频稳定**框架，不依赖成对的稳定 / 不稳定视频标签，并且在推理过程中不使用未来帧 look-ahead 信息。该方法以经典视频稳定流程为基础，结合实用先验与多线程缓冲机制，面向因果在线部署和资源受限硬件场景。

本工作受到了 [DUTCode](https://github.com/Annbless/DUTCode.git) 的巨大启发。我们非常感谢原作者的无私开源，并向其优秀工作致敬。截至目前，我们仍然认为 **DUTCode 是最优秀的视频稳定方法之一**。

---

## 海报与视频

- [CVPR 2026 虚拟海报页面](https://cvpr.thecvf.com/virtual/2026/poster/39183)
- [arXiv 页面](https://arxiv.org/abs/2602.23141)
- [论文 PDF](https://arxiv.org/pdf/2602.23141)
- [YouTube 报告视频](https://www.youtube.com/watch?v=SBrtgR3HAJo)

<!--
如果 CVPR 托管的海报图在虚拟站点更新完成前无法渲染，
可以把下面的 src 替换为本仓库中的本地文件，例如 assets/poster.png。
GitHub README 支持相对路径图片。
-->

<p align="center">
  <a href="https://cvpr.thecvf.com/virtual/2026/poster/39183">
    <img src="https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202026/39183.png" alt="CVPR 2026 Poster" width="95%">
  </a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=SBrtgR3HAJo">
    <img src="https://img.youtube.com/vi/SBrtgR3HAJo/maxresdefault.jpg" alt="观看 CVPR 2026 报告视频" width="70%">
  </a>
</p>

---

## 项目特点

- **无监督** 视频稳定
- **在线视频推理**，无需未来帧 look-ahead
- 基于有效的 **经典先验**
- 面向实时 / 在线场景的轻量实用改造
- 多线程缓冲机制，支持高效因果处理
- 面向多模态无人机航拍视频稳定的 **UAV-Test** 测试集

---

## 方法说明

本方法聚焦于**无标签、无未来帧信息条件下的无监督在线视频稳定**。

与依赖未来帧信息的离线稳定方法不同，本工作更关注真实部署中更实用的在线场景，即每一帧都以因果方式处理。整体框架遵循经典视频稳定流程，并结合运动估计、运动传播与运动补偿中的实用先验。

该设计主要针对端到端学习稳定方法中的三个实际问题：

1. 成对稳定 / 不稳定训练数据难以获取；
2. 完全学习式流程的可控性较弱；
3. 在资源受限硬件上的部署效率不足。

更多技术细节、可视化结果和消融实验将在仓库整理后逐步补充。

---

## 当前状态

- 论文：**CVPR 2026 已接收**
- arXiv 预印本：**已开放**
- CVPR 虚拟海报页面：**已开放**
- YouTube 报告视频：**已开放**
- 推理 / 在线演示：**已提供**
- 测试数据集（**UAV-Test**）：**已提供**
- 训练脚本：**暂未开放**
- 完整训练数据集：**即将开放**

---

## 安装方法

### 1. 下载资源文件

请先下载 `LightStab_assets` 文件，并将其中**全部内容直接复制到项目根目录**。

这些文件主要包括预训练权重和运行所需资源。

[下载 LightStab_assets](https://drive.usercontent.google.com/download?id=1pHD3BR2KXKHjksKTx5z50HAE-2GNOO17&export=download&authuser=0&confirm=t&uuid=cd5409a4-0e8e-49f4-8189-23ef4f6ea6c1&at=AGN2oQ07Ev9BOwUa2gHbhyxK3fr3:1774845474015)

### 2. 创建并激活环境

```bash
conda env create -n lightstab -f environment.yaml
conda activate lightstab
```

---

## 快速开始

运行在线视频稳定脚本：

```bash
cd scripts
python onlinestab.py
```

---

## 项目结构

一个典型的项目结构如下：

```text
.
├── scripts/
│   └── onlinestab.py
├── environment.yaml
├── README.md
└── ...
```

随着后续代码整理和更新，具体目录结构可能还会发生变化。

---

## 待办事项

- [x] 发布初版推理代码
- [x] 提供运行所需权重与资源文件
- [x] 发布测试数据集（**UAV-Test**）
- [x] 添加 CVPR 虚拟海报页面链接
- [x] 添加 YouTube 报告视频链接
- [ ] 发布整理后的完整仓库版本
- [ ] 发布完整训练数据集
- [ ] 发布训练脚本
- [ ] 补充定性可视化结果
- [ ] 补充更详细的使用文档
- [ ] 更新期刊扩展版本相关内容

---

## 重要说明

由于我们正在准备该工作的期刊扩展版本，因此**训练脚本暂时不会公开**。

我们预计将在完成扩刊与投稿流程后，于**8 月底左右**公开**完整训练代码和数据集**。

在此之前，**请不要通过邮件索要训练脚本**，相关请求将不再单独回复。

感谢理解与支持。

---

## 致谢

我们感谢以下优秀的开源仓库和项目对本工作的启发：

- [Grundmann et al.](https://github.com/ishank-juneja/L1-optimal-paths-Stabilization.git)
- [Bundle](https://github.com/SuTanTank/BundledCameraPathVideoStabilization.git)
- [DIFRINT](https://github.com/jinsc37/DIFRINT.git)
- [Yu and Ramamoorthi](https://jiyang.fun/projects.html)
- [PWStableNet](https://github.com/mindazhao/PWStableNet.git)
- [DUT](https://github.com/Annbless/DUTCode.git)
- [Deep3D](https://github.com/yaochih/Deep3D-Stabilizer-release.git)
- [FuSta](https://github.com/alex04072000/FuSta.git)
- [RStab](https://github.com/pzzz-cv/RStab.git)
- [MetaStab](https://github.com/MKashifAli/MetaVideoStab.git)
- [GaVS](https://github.com/huawei-bayerlab/GaVS.git)
- [MeshFlow](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization.git)
- [StabNet](https://github.com/cxjyxxme/deep-online-video-stabilization-deploy.git)
- [NNDVS](https://github.com/liuzhen03/NNDVS.git)
- [Liu et al.](https://github.com/liutao23/Realtime_Video_Stabilization.git)

特别感谢 **DUTCode** 的作者们，他们的开源工作给了我们很大启发。

---

## 引用

如果你觉得本仓库对你的研究有帮助，欢迎引用以下工作：

```bibtex
@inproceedings{liu2026nolabelsnolookahead,
  title={No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors},
  author={Liu, Tao and Wan, Gang and Ren, Kan and Wen, Shibo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

@article{xu2022dut,
  title={Dut: Learning video stabilization by simply watching unstable videos},
  author={Xu, Yufei and Zhang, Jing and Maybank, Stephen J. and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={4306--4320},
  year={2022},
  publisher={IEEE}
}
```

arXiv 版本见 [arXiv:2602.23141](https://arxiv.org/abs/2602.23141)。

---

## 常见问题

### Q1：训练脚本现在开放了吗？

暂时没有。训练脚本会在期刊扩展和投稿流程完成后公开。

### Q2：现在可以下载数据集吗？

目前仅开放论文中的测试数据集 **UAV-Test**：

[下载 UAV-Test](https://drive.usercontent.google.com/download?id=1eHJ4Z8uqPKheDDzea4nQWGwwq5aauBvh&export=download&authuser=0&confirm=t&uuid=f05ae258-5403-4dec-add8-27ce19a96fab&at=AGN2oQ1ce8ktgFS0Qhf1KN_8P4dF:1775119978751)

训练数据并不是原始视频帧，而是提取后的关键点和光流数据。我们还需要一些时间来整理这些内容，并与完整数据集一起发布。

### Q3：可以通过邮件提前索要训练脚本吗？

目前请不要邮件索要，相关请求将不再单独回复。
