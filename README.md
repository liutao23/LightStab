# No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors

> [CVPR 2026] Unsupervised online video stabilization with classical priors, without labels and without look-ahead.

[English](#english) | [中文](#中文)

---

## English

### Overview

This repository contains the official implementation of:

**No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors**

We are currently revising and polishing the manuscript.  
The code and datasets will be released soon.

This project is heavily inspired by [DUTCode](https://github.com/Annbless/DUTCode.git).  
We sincerely thank the original authors for their generous open-source contribution and pay tribute to their excellent work.

Our contribution mainly consists of **small but practical modifications** that make the framework suitable for **online video stabilization**.  
Up to now, we still believe that **DUTCode remains one of the best video stabilization methods**.

---

### Highlights

- **Unsupervised** video stabilization
- **Online** inference without look-ahead
- Built upon strong **classical priors**
- Lightweight adaptation from previous excellent stabilization frameworks
- Practical deployment-oriented design

---

### Project Status

- Manuscript: **under revision**
- Inference / online demo: **available**
- Training scripts: **not released yet**
- Dataset: **coming soon**

---

### Installation

#### 1. Download assets

Please download the `LightStab_assets` package and copy **all files** directly into the project root directory.

These files include pretrained weights and other required assets.

Download link:  
https://drive.usercontent.google.com/download?id=1pHD3BR2KXKHjksKTx5z50HAE-2GNOO17&export=download&authuser=0&confirm=t&uuid=cd5409a4-0e8e-49f4-8189-23ef4f6ea6c1&at=AGN2oQ07Ev9BOwUa2gHbhyxK3fr3:1774845474015

#### 2. Create and activate the environment

```bash
conda env create -n lightstab -f environment.yaml
conda activate lightstab
```

#### 3. Run online stabilization

```bash
cd scripts
python onlinestab.py
```

---

### Important Note

We are currently preparing an extended journal version of this work.  
For this reason, the **training scripts are not publicly available at this stage**.

We expect to release the **full training code and dataset** after the journal extension and submission process is completed, which is currently planned for **around the end of August**.

Before that, **please do not email us to request the training scripts**, as we will not be able to respond to such requests.

Thank you for your understanding and support.

---

### Acknowledgements

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

---

### Citation

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

@article{Liu_Wan_Bai_Kong_Tang_Wang_2024,
  title={Real-Time Video Stabilization Algorithm Based on SuperPoint},
  volume={73},
  DOI={10.1109/tim.2023.3342849},
  journal={IEEE Transactions on Instrumentation and Measurement},
  author={Liu, Tao and Wan, Gang and Bai, Hongyang and Kong, Xiaofang and Tang, Bo and Wang, Fangyi},
  year={2024},
  month={Jan},
  pages={1--13},
  language={English}
}
```

---

## 中文

### 项目简介

本仓库为以下论文的官方实现：

**No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors**

目前我们正在对论文稿件进行进一步修改和润色。  
代码与数据集将在近期公开。

本工作受到了 [DUTCode](https://github.com/Annbless/DUTCode.git) 的巨大启发。  
我们非常感谢原作者的无私开源，并向其优秀工作致敬。

本仓库的主要工作是在已有优秀方法基础上做了**小幅但实用的改进**，重点是让方法能够支持**在线视频稳定**。  
截至目前，我们仍然认为 **DUTCode 是最优秀的视频稳定方法之一**。

---

### 项目特点

- **无监督** 视频稳定
- **在线视频稳定**，无需 look-ahead
- 融合有效的 **经典先验**
- 在已有优秀工作基础上进行轻量改造
- 更面向实际部署场景

---

### 当前状态

- 论文：**正在修改中**
- 推理 / 在线演示：**已提供**
- 训练脚本：**暂未开放**
- 数据集：**即将开放**

---

### 安装方法

#### 1. 下载资源文件

请先下载 `LightStab_assets` 文件，并将其中**全部内容直接复制到项目根目录**。

这些文件主要包括预训练权重及相关资源。

下载链接：  
https://drive.usercontent.google.com/download?id=1pHD3BR2KXKHjksKTx5z50HAE-2GNOO17&export=download&authuser=0&confirm=t&uuid=cd5409a4-0e8e-49f4-8189-23ef4f6ea6c1&at=AGN2oQ07Ev9BOwUa2gHbhyxK3fr3:1774845474015

#### 2. 创建并激活环境

```bash
conda env create -n lightstab -f environment.yaml
conda activate lightstab
```

#### 3. 运行在线稳定脚本

```bash
cd scripts
python onlinestab.py
```

---

### 重要说明

由于我们正在准备该工作的期刊扩展版本，  
因此**训练脚本暂时不会公开**。

我们预计将在**今年 8 月底左右**，完成扩刊与投稿流程后，立即公开**完整训练代码和数据集**。

在此之前，**请不要通过邮件索要训练脚本**，相关邮件将不再单独回复。  
感谢理解与支持。

---

### 致谢

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

---

### 引用

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

@article{Liu_Wan_Bai_Kong_Tang_Wang_2024,
  title={Real-Time Video Stabilization Algorithm Based on SuperPoint},
  volume={73},
  DOI={10.1109/tim.2023.3342849},
  journal={IEEE Transactions on Instrumentation and Measurement},
  author={Liu, Tao and Wan, Gang and Bai, Hongyang and Kong, Xiaofang and Tang, Bo and Wang, Fangyi},
  year={2024},
  month={Jan},
  pages={1--13},
  language={English}
}
```
