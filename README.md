# No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors

<p align="center">
  <b>[CVPR 2026]</b><br>
  Unsupervised Online Video Stabilization with Classical Priors
</p>

<p align="center">
  <a href="#english">English</a> •
  <a href="#中文">中文</a>
</p>

---

## Table of Contents

- [English](#english)
  - [News](#news)
  - [Overview](#overview)
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

- **[2026-03]** Repository initialized.
- **[2026-03]** Inference code for online stabilization is available.
- **[Coming Soon]** Code cleanup, dataset release, and training scripts.

---

## Overview

This repository presents our work:

**No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors**

We are currently revising and polishing the manuscript.  
The code and datasets will be released soon.

This work is heavily inspired by [DUTCode](https://github.com/Annbless/DUTCode.git).  
We sincerely thank the original authors for their generous open-source contribution and pay tribute to their excellent work.

Our work mainly introduces **practical modifications** to support **online video stabilization**, while keeping the spirit of strong prior-based stabilization methods.  
Up to now, we still believe that **DUTCode remains one of the best video stabilization methods**.

---

## Highlights

- **Unsupervised** video stabilization
- **Online** inference without look-ahead
- Built upon strong **classical priors**
- Practical and lightweight adaptation for real-time / online settings
- Inspired by strong existing stabilization frameworks

---

## Method

Our method focuses on **online video stabilization without labels and without future-frame look-ahead**.

Unlike offline stabilization methods that rely on future information, this work targets a more practical online setting, where each frame is processed causally.  
The framework is designed with classical priors in mind and inherits strong inspiration from prior stabilization literature.

> More technical details, visual results, and ablation studies will be added after the manuscript revision is finalized.

---

## Project Status

- Manuscript: **under revision**
- Inference / online demo: **available**
- Training scripts: **not released yet**
- Dataset: **coming soon**

---

## Installation

### 1. Download assets

Please download the `LightStab_assets` package and copy **all files** directly into the project root directory.

These files include pretrained weights and required runtime assets.

Download link:  
https://drive.usercontent.google.com/download?id=1pHD3BR2KXKHjksKTx5z50HAE-2GNOO17&export=download&authuser=0&confirm=t&uuid=cd5409a4-0e8e-49f4-8189-23ef4f6ea6c1&at=AGN2oQ07Ev9BOwUa2gHbhyxK3fr3:1774845474015

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

> The exact structure may evolve as the repository is further cleaned and updated.

---

## TODO

- [x] Release initial inference code
- [x] Provide runtime assets / pretrained weights
- [ ] Release cleaned repository version
- [ ] Release dataset
- [ ] Release training scripts
- [ ] Add qualitative visualization results
- [ ] Add more detailed documentation
- [ ] Add journal-extension version updates

---

## Important Note

We are currently preparing an extended journal version of this work.  
For this reason, the **training scripts are not publicly available at this stage**.

We expect to release the **full training code and dataset** after the journal extension and submission process is completed, which is currently planned for **around the end of August**.

Before that, **please do not email us to request the training scripts**, as we will not be able to respond to such requests.

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
@article{xu2022dut,
  title={Dut: Learning video stabilization by simply watching unstable videos},
  author={Xu, Yufei and Zhang, Jing and Maybank, Stephen J and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={4306--4320},
  year={2022},
  publisher={IEEE}
}
@article{liu2026no,
  title={No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors},
  author={Liu, Tao and Wan, Gang and Ren, Kan and Wen, Shibo},
  journal={arXiv preprint arXiv:2602.23141},
  year={2026}
}

```

---

## FAQ

### Q1: Are the training scripts available?

Not yet. We plan to release them after the journal extension and submission process is completed.

### Q2: Is the dataset available now?

Not yet. It will be released together with the complete training resources.

### Q3: Can I request the training scripts by email?

Please do not email us for training scripts at this stage. Such requests will not be replied to.

---

# 中文

## 更新日志

- **[2026-03]** 初始化仓库。
- **[2026-03]** 提供在线视频稳定推理代码。
- **[即将发布]** 代码整理、数据集与训练脚本。

---

## 项目简介

本仓库对应论文：

**No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors**

目前我们正在对论文稿件进行进一步修改和润色。  
代码与数据集将在近期公开。

本工作受到了 [DUTCode](https://github.com/Annbless/DUTCode.git) 的巨大启发。  
我们非常感谢原作者的无私开源，并向其优秀工作致敬。

本仓库主要是在现有优秀方法基础上进行了**小幅但实用的改进**，核心目标是支持**在线视频稳定**。  
截至目前，我们仍然认为 **DUTCode 是最优秀的视频稳定方法之一**。

---

## 项目特点

- **无监督** 视频稳定
- **在线视频稳定**，无需 look-ahead
- 融合有效的 **经典先验**
- 面向在线/实时场景的轻量改造
- 基于已有优秀稳定框架进一步扩展

---

## 方法说明

本方法聚焦于**无标签、无未来帧信息条件下的在线视频稳定**。

与依赖未来帧信息的离线稳定方法不同，本工作更关注真实部署中更实用的在线场景，即每一帧都以因果方式处理。  
整体框架保留了经典先验方法的思想，并吸收了大量已有视频稳定工作的启发。

> 更多技术细节、可视化结果和消融实验将在论文修改完成后逐步补充。

---

## 当前状态

- 论文：**正在修改中**
- 推理 / 在线演示：**已提供**
- 训练脚本：**暂未开放**
- 数据集：**即将开放**

---

## 安装方法

### 1. 下载资源文件

请先下载 `LightStab_assets` 文件，并将其中**全部内容直接复制到项目根目录**。

这些文件主要包括预训练权重和运行所需资源。

下载链接：  
https://drive.usercontent.google.com/download?id=1pHD3BR2KXKHjksKTx5z50HAE-2GNOO17&export=download&authuser=0&confirm=t&uuid=cd5409a4-0e8e-49f4-8189-23ef4f6ea6c1&at=AGN2oQ07Ev9BOwUa2gHbhyxK3fr3:1774845474015

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

> 随着后续代码整理和更新，具体目录结构可能还会发生变化。

---

## 待办事项

- [x] 发布初版推理代码
- [x] 提供运行所需权重与资源文件
- [ ] 发布整理后的完整仓库版本
- [ ] 发布数据集
- [ ] 发布训练脚本
- [ ] 补充定性可视化结果
- [ ] 补充更详细的使用文档
- [ ] 更新期刊扩展版本相关内容

---

## 重要说明

由于我们正在准备该工作的期刊扩展版本，  
因此**训练脚本暂时不会公开**。

我们预计将在**今年 8 月底左右**，完成扩刊与投稿流程后，立即公开**完整训练代码和数据集**。

在此之前，**请不要通过邮件索要训练脚本**，相关邮件将不再单独回复。  
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
@article{xu2022dut,
  title={Dut: Learning video stabilization by simply watching unstable videos},
  author={Xu, Yufei and Zhang, Jing and Maybank, Stephen J and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={4306--4320},
  year={2022},
  publisher={IEEE}
}
@article{liu2026no,
  title={No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors},
  author={Liu, Tao and Wan, Gang and Ren, Kan and Wen, Shibo},
  journal={arXiv preprint arXiv:2602.23141},
  year={2026}
}
```

---

## 常见问题

### Q1：训练脚本现在开放了吗？

暂时没有。训练脚本会在期刊扩展和投稿流程完成后公开。

### Q2：数据集现在可以下载吗？

暂时还不可以。后续会和完整训练资源一起发布。

### Q3：可以通过邮件提前索要训练脚本吗？

目前请不要邮件索要，相关请求将不再单独回复。
