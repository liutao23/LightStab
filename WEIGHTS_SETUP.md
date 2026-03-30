# LightStab 权重与大文件放置说明

本项目的 Git 仓库只保留代码，不直接包含模型权重、训练数据和其他大文件。
请先从作者提供的网盘下载这些文件，然后按下面的路径放回项目目录。

## 读者使用步骤

1. 克隆代码仓库。
2. 从网盘下载权重压缩包或单独文件。
3. 将下载得到的文件放到下面列出的**目标路径**。
4. 放好后，运行校验命令：

```bash
python separate_assets.py verify --repo . --manifest weights_manifest.json
```

如果输出 `所有文件都存在`，说明路径基本正确。

## 推荐目录结构

下面这些路径是文件应该放回项目中的位置：

| 文件名 | 目标路径 | 大小 | SHA256 |
|---|---|---:|---|
| twins_svt_large-90f6aaa9.pth | `OffTheShelfModule/optical_module/core/weights/twins_svt_large-90f6aaa9.pth` | 378.75 MB | `90f6aaa970af86bdb15111d45bb3a3a4b3dac4a38fd267d08d31402dd540bb80` |
| eloftr_outdoor.ckpt | `OffTheShelfModule/point_module/weight/eloftr_outdoor.ckpt` | 183.87 MB | `0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878` |
| ProPainter.pth | `OffTheShelfModule/outpainting/weights/ProPainter.pth` | 150.47 MB | `12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591` |
| kitti.pth | `OffTheShelfModule/optical_module/core/weights/kitti.pth` | 62.00 MB | `8ea3ad4a989827545fd788a616de66bf7f82ce5586e79a7762fe3dea957d1f71` |
| sintel.pth | `OffTheShelfModule/optical_module/core/weights/sintel.pth` | 62.00 MB | `3526b42eb2ce374cf1c035c556fec0edf47de4ddc91c6e400925443610bbf16c` |
| MemFlowNet_T_things_kitti.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_T_things_kitti.pth` | 48.76 MB | `c6a27af787dbe43f004419cee0a0dcff6f2ce0144ac908f6b56da291f008d19f` |
| MemFlowNet_T_sintel.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_T_sintel.pth` | 48.76 MB | `89048d5af5152ecd46fbafc21e41375ee5c66c2816719b31263fe50b10764f3f` |
| MemFlowNet_T_things.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_T_things.pth` | 48.76 MB | `2f874ce842419a28a2aef667876ff8a9bdeac7a4f103aef85447c2c954b3ba3c` |
| MemFlowNet_T_kitti.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_T_kitti.pth` | 48.76 MB | `5fa86bdc6b3e899902e58a31675deb22e88b21c109b90bc9140e3d4ce5545cd9` |
| twins_skflow.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/twins_skflow.pth` | 48.74 MB | `e51601cfb25ee235b7314b6e2579303b090cf9a41788fcb2370bebcdb7a0c8aa` |
| doghardnet_lightglue.pth | `OffTheShelfModule/point_module/weight/doghardnet_lightglue.pth` | 45.43 MB | `21468c1b417a1b15f83a7c3b5ae2bd5e1d7c31d392aa2732e247e7223bb4aa40` |
| aliked_lightglue.pth | `OffTheShelfModule/point_module/weight/aliked_lightglue.pth` | 45.43 MB | `d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503` |
| sift_lightglue.pth | `OffTheShelfModule/point_module/weight/sift_lightglue.pth` | 45.43 MB | `5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0` |
| disk_lightglue.pth | `OffTheShelfModule/point_module/weight/disk_lightglue.pth` | 45.42 MB | `b5b21d47ea24f2c5e501aec9c91b9716e4c8c3429a4dc1e615c133c4c9378335` |
| superpoint_lightglue.pth | `OffTheShelfModule/point_module/weight/superpoint_lightglue.pth` | 45.30 MB | `6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a` |
| neuflow_things.pth | `OffTheShelfModule/optical_module/NeuFlow/neuflow_things.pth` | 34.52 MB | `733d13b1b2202adefcc99bd1f0fceb89fc90da5479f9826fa3f17ff42c4bdbe0` |
| neuflow_mixed.pth | `OffTheShelfModule/optical_module/NeuFlow/neuflow_mixed.pth` | 34.52 MB | `76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f` |
| neuflow_sintel.pth | `OffTheShelfModule/optical_module/NeuFlow/neuflow_sintel.pth` | 34.52 MB | `9bc12c9ef8298e3cca08a33a04bf99b63dc989e16808daca0c10ac5be4291eb1` |
| hardnet8v2.pt | `OffTheShelfModule/point_module/weight/hardnet8v2.pt` | 34.49 MB | `5982f6d2c647545f13e74e8b011110fea2408130015a1496713253d9c00a941b` |
| dad.pth | `OffTheShelfModule/point_module/weight/dad.pth` | 24.87 MB | `7d827844649faf5213ba97e7adb43e16b936dd3d11ab2d0fc0d602c67fae781f` |
| MemFlowNet_sintel.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_sintel.pth` | 24.22 MB | `47c0ecd567612e518018a278dae5b2cfae5c61d2c720e44dfb63b20cad91daf3` |
| MemFlowNet_spring.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_spring.pth` | 24.22 MB | `c0cc9ed3099eea14bcff0090d0d0d362a34ccc2214a5526c3a6690d8b545880e` |
| MemFlowNet_kitti.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_kitti.pth` | 24.22 MB | `c3ecadb24486ad04eaeeb153f88620fbf931c69b3dab48a8f32ef4d6329fef5c` |
| skflow-things.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/skflow-things.pth` | 24.22 MB | `a1caa4a1a406e0831188f444ef03822014e440f7ef16eeb965a0b5f08ba5f34b` |
| raft-things.pth | `OffTheShelfModule/outpainting/weights/raft-things.pth` | 20.13 MB | `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1` |
| raft_large_C_T_SKHT_V2-ff5fadd5.pth | `OffTheShelfModule/optical_module/raft_weight/raft_large_C_T_SKHT_V2-ff5fadd5.pth` | 20.13 MB | `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322` |
| MemFlowNet_P_things.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_P_things.pth` | 19.88 MB | `aae711d3bff6eb79bbd9fdaacb233ae41c5d99359d3fb7cd68583a140e09a8d7` |
| MemFlowNet_P_sintel.pth | `OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_P_sintel.pth` | 19.88 MB | `4137fa814ab36757b1b9929bf42eed4a71543a7ad9efccff8c7c69102fc2a7fa` |
| recurrent_flow_completion.pth | `OffTheShelfModule/outpainting/weights/recurrent_flow_completion.pth` | 19.41 MB | `22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283` |
| xfeat.pt | `OffTheShelfModule/point_module/modules/xfeat.pt` | 5.96 MB | `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b` |
| RFDet_weights.pth | `OffTheShelfModule/point_module/weight/RFDet_weights.pth` | 5.22 MB | `a87b33070d5c5fa20c5d2ba17924bbd6b219715950340f31bf03a1eecc488d13` |
| HardNet++.pth | `OffTheShelfModule/point_module/weight/HardNet++.pth` | 5.10 MB | `9fe80a335fe72d1225fdd1ae0943418b12489f95014bd51dddc9819d6ffa33c7` |
| checkpoint_liberty_with_aug.pth | `OffTheShelfModule/point_module/weight/checkpoint_liberty_with_aug.pth` | 5.10 MB | `1e9a41b19f1dc93c986e91df9aaf5696d1a777ac1d67498492856a65d6f49c16` |
| superpoint_v1.pth | `OffTheShelfModule/point_module/weight/superpoint_v1.pth` | 4.96 MB | `52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e` |
| epipolar-save.pth | `OffTheShelfModule/point_module/weight/epipolar-save.pth` | 4.17 MB | `6e0cf5df4abab1fddbf0b62325647c9a6f387453b28c4b986542b7dabda972ee` |
| depth-save.pth | `OffTheShelfModule/point_module/weight/depth-save.pth` | 4.17 MB | `9c2ee4ded238892dfa51569941372601e35e4a74aa6f84ea80053d2ab1c07abe` |
| raft_small_C_T_V2-01064c6d.pth | `OffTheShelfModule/optical_module/raft_weight/raft_small_C_T_V2-01064c6d.pth` | 3.82 MB | `01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27` |
| aliked-n32.pth | `OffTheShelfModule/point_module/weight/aliked-n32.pth` | 3.76 MB | `055283fd1624d3ab46a05c3b617b1a79a3f28d1f551737eef2d653e8c8232890` |
| aliked-n16rot.pth | `OffTheShelfModule/point_module/weight/aliked-n16rot.pth` | 2.61 MB | `ddf3abbf38e86f6a74540d214e1a9712c54b2d8551abc864542199f2347d7332` |
| aliked-n16.pth | `OffTheShelfModule/point_module/weight/aliked-n16.pth` | 2.61 MB | `5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2` |
| aliked-t16.pth | `OffTheShelfModule/point_module/weight/aliked-t16.pth` | 776.23 KB | `56782336919db220144a884e710e2b6af13b3b8f8da15b915a239078ab471085` |
| EfficientMotionPro.pth | `preweights/EfficientMotionPro.pth` | 133.68 KB | `267f20a2176542e9dfa89db14dc7a6b95e654791ce85ae89eeb562da8511406f` |
| PropagationModel_best_att_all_0.0002properMulti_oldloss.pth | `weights/Motion/PropagationModel_best_att_all_0.0002properMulti_oldloss.pth` | 133.25 KB | `14cb07c8ef96906b9431bcb5707473ab120e4e3986ca421c2837a960f3bc5ffa` |
| PropagationModel_ch_att_all_0.0002properMulti_oldloss.pth | `weights/Motion/PropagationModel_ch_att_all_0.0002properMulti_oldloss.pth` | 131.65 KB | `61ff5570216f30572c200a227a6e31771123b6af3847163e645e5c3b39f8d1fd` |
| PropagationModellarge_best_att_all_0.0002properMulti_oldloss.pth | `weights/Motionnotemp/PropagationModellarge_best_att_all_0.0002properMulti_oldloss.pth` | 129.91 KB | `943b368ca3190eecd8ed446b49d1f52d0d6b9ec345f9cdba5188b059a82867a4` |
| PropagationModellarge_ch_att_all_0.0002properMulti_oldloss.pth | `weights/Motionnotemp/PropagationModellarge_ch_att_all_0.0002properMulti_oldloss.pth` | 129.49 KB | `bfb9ec6f856092da5d91e51bea4c83b778009c8c42662d3905e8b07738bfd9a1` |
| LightOnlineSmoother.pth | `preweights/LightOnlineSmoother.pth` | 18.98 KB | `94baa7313a442747a748a5134f70f7b48e18de43c00a9e7e99920876bfd69ef7` |
| SmootherModel_best_all_2e-05LossTypeL1.pth | `weights/SmootherNokp/SmootherModel_best_all_2e-05LossTypeL1.pth` | 18.87 KB | `5b01a6d5476f8a78a438fdf635dda2a9229350df24f6dae0f92de3f70eda4959` |
| SmootherModel_best_all_2e-05LossTypeL1.pth | `weights/Smoother_5/SmootherModel_best_all_2e-05LossTypeL1.pth` | 18.87 KB | `88d48df841525064ce6c131793efc6969ed43cf517b4d4040ed9a60306e986d9` |
| SmootherModel_best_all_2e-05LossTypeL1.pth | `weights/Smoother/SmootherModel_best_all_2e-05LossTypeL1.pth` | 18.87 KB | `81eff5e65ba9d1cdbcf3b02884c38f3f29cb04239719e7f7947e9ed7c3919627` |
| SmootherModel_best_all_2e-05LossTypeL1.pth | `weights/Smoother_9/SmootherModel_best_all_2e-05LossTypeL1.pth` | 18.87 KB | `b1f268625508ec70ba290a6f251a3dc0364155c2448029dbe9db541d211971c2` |
| SmootherModel_ch_all_2e-05LossTypeL1.pth | `weights/SmootherNokp/SmootherModel_ch_all_2e-05LossTypeL1.pth` | 18.82 KB | `2ba4d1197adc7220cfa2fd45f1f1b90115acdeafd22a139969fc36d0e0b0e33c` |
| SmootherModel_ch_all_2e-05LossTypeL1.pth | `weights/Smoother_5/SmootherModel_ch_all_2e-05LossTypeL1.pth` | 18.82 KB | `2af4d017fb2c22e110aa6cf1fd0e8d5815f21800783bb13fd03470b9dc2a32d1` |
| SmootherModel_ch_all_2e-05LossTypeL1.pth | `weights/Smoother/SmootherModel_ch_all_2e-05LossTypeL1.pth` | 18.82 KB | `e0c1323a00da4d9df4b3c34dd70d5a4df72470a4d4d7a38226c4abf68f1bd70a` |
| SmootherModel_ch_all_2e-05LossTypeL1.pth | `weights/Smoother_9/SmootherModel_ch_all_2e-05LossTypeL1.pth` | 18.82 KB | `5726fb76e60e68c9eaf7da96cca8710c5e633b58c551851453bf16419b096d38` |

## 建议的网盘目录组织

为了让读者更容易还原，建议你在网盘里保持与仓库相同的目录结构，例如：

```text
OffTheShelfModule/optical_module/core/weights/twins_svt_large-90f6aaa9.pth
OffTheShelfModule/point_module/weight/eloftr_outdoor.ckpt
OffTheShelfModule/outpainting/weights/ProPainter.pth
OffTheShelfModule/optical_module/core/weights/kitti.pth
OffTheShelfModule/optical_module/core/weights/sintel.pth
OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_T_things_kitti.pth
OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_T_sintel.pth
OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_T_things.pth
OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_T_kitti.pth
OffTheShelfModule/optical_module/Memflow/ckpts/twins_skflow.pth
OffTheShelfModule/point_module/weight/doghardnet_lightglue.pth
OffTheShelfModule/point_module/weight/aliked_lightglue.pth
OffTheShelfModule/point_module/weight/sift_lightglue.pth
OffTheShelfModule/point_module/weight/disk_lightglue.pth
OffTheShelfModule/point_module/weight/superpoint_lightglue.pth
OffTheShelfModule/optical_module/NeuFlow/neuflow_things.pth
OffTheShelfModule/optical_module/NeuFlow/neuflow_mixed.pth
OffTheShelfModule/optical_module/NeuFlow/neuflow_sintel.pth
OffTheShelfModule/point_module/weight/hardnet8v2.pt
OffTheShelfModule/point_module/weight/dad.pth
OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_sintel.pth
OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_spring.pth
OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_kitti.pth
OffTheShelfModule/optical_module/Memflow/ckpts/skflow-things.pth
OffTheShelfModule/outpainting/weights/raft-things.pth
OffTheShelfModule/optical_module/raft_weight/raft_large_C_T_SKHT_V2-ff5fadd5.pth
OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_P_things.pth
OffTheShelfModule/optical_module/Memflow/ckpts/MemFlowNet_P_sintel.pth
OffTheShelfModule/outpainting/weights/recurrent_flow_completion.pth
OffTheShelfModule/point_module/modules/xfeat.pt
OffTheShelfModule/point_module/weight/RFDet_weights.pth
OffTheShelfModule/point_module/weight/HardNet++.pth
OffTheShelfModule/point_module/weight/checkpoint_liberty_with_aug.pth
OffTheShelfModule/point_module/weight/superpoint_v1.pth
OffTheShelfModule/point_module/weight/epipolar-save.pth
OffTheShelfModule/point_module/weight/depth-save.pth
OffTheShelfModule/optical_module/raft_weight/raft_small_C_T_V2-01064c6d.pth
OffTheShelfModule/point_module/weight/aliked-n32.pth
OffTheShelfModule/point_module/weight/aliked-n16rot.pth
OffTheShelfModule/point_module/weight/aliked-n16.pth
OffTheShelfModule/point_module/weight/aliked-t16.pth
preweights/EfficientMotionPro.pth
weights/Motion/PropagationModel_best_att_all_0.0002properMulti_oldloss.pth
weights/Motion/PropagationModel_ch_att_all_0.0002properMulti_oldloss.pth
weights/Motionnotemp/PropagationModellarge_best_att_all_0.0002properMulti_oldloss.pth
weights/Motionnotemp/PropagationModellarge_ch_att_all_0.0002properMulti_oldloss.pth
preweights/LightOnlineSmoother.pth
weights/SmootherNokp/SmootherModel_best_all_2e-05LossTypeL1.pth
weights/Smoother_5/SmootherModel_best_all_2e-05LossTypeL1.pth
weights/Smoother/SmootherModel_best_all_2e-05LossTypeL1.pth
weights/Smoother_9/SmootherModel_best_all_2e-05LossTypeL1.pth
weights/SmootherNokp/SmootherModel_ch_all_2e-05LossTypeL1.pth
weights/Smoother_5/SmootherModel_ch_all_2e-05LossTypeL1.pth
weights/Smoother/SmootherModel_ch_all_2e-05LossTypeL1.pth
weights/Smoother_9/SmootherModel_ch_all_2e-05LossTypeL1.pth
```

这样读者下载后可以直接解压到项目根目录。

## 给作者的建议

- 仓库里只保留代码、配置文件、示例脚本和 README。
- 权重文件放在网盘，并在主 README 中添加下载链接。
- 不要把这些大文件重新提交到 Git。
- 如果以后要在 GitHub 管理大文件，可以改用 Git LFS。

## 自动生成说明

此文档由 `separate_assets.py` 根据 `weights_manifest.json` 自动生成。
