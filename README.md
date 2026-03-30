# [CVPR 2026] No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors

We are in the process of revising and polishing the manuscript. The code and datasets will be made available shortly.
本工作受到了DUTCode（https://github.com/Annbless/DUTCode.git）的巨大启发，也是非常感谢原作者的免费开源，我们向他致敬！
做的工作是小修小改，主要是让其在线运行！截止目前为止，我仍然认为DUTCode是最好的视频稳定方法。

1.下载LightStab_assets文件，直接将内容全都复制到根目录下，这是一些权重文件。
下载链接：
2.创建并激活环境：conda env create -n lightstab -f environment.yaml conda activate lightstab 
3.cd scripts python onlinestab.py
4.需要注意，由于准备将论文扩刊到期刊上面，所以目前我们尚未开放训练脚本，但预计今年8月底份，我们扩刊完毕投稿完成会立刻一并释放全部训练代码和数据集。在此之前请不要向我们发送邮件索要训练脚本，我们不会回复。
我们感谢下面很棒的仓库对本工作的启发
https://github.com/Annbless/DUTCode.git



如果觉得本仓库有用，请引用

@article{liu2026no,
  title={No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors},
  author={Liu, Tao and Wan, Gang and Ren, Kan and Wen, Shibo},
  journal={arXiv preprint arXiv:2602.23141},
  year={2026}
}
@article{xu2022dut,
  title={Dut: Learning video stabilization by simply watching unstable videos},
  author={Xu, Yufei and Zhang, Jing and Maybank, Stephen J and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={4306--4320},
  year={2022},
  publisher={IEEE}
}
