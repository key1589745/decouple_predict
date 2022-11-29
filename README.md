# Decoupling Predictions in Distributed Learning for Multi-Center Left Atrial MRI Segmentation
This project is developed for the MICCAI 2022 paper: [Decoupling Predictions in Distributed Learning
for Multi-Center Left Atrial MRI Segmentation](https://arxiv.org/abs/2206.05284v1). Our code is implemented based on the [Learn_Noisy_Labels_Medical_Images](https://github.com/moucheng2017/Learn_Noisy_Labels_Medical_Images) and [probabilistic_unet](https://github.com/SimonKohl/probabilistic_unet), but we used them to tackle the non-IID challenge in distributed leearning. For more information, please read the following paper:

<div align=center><img src="framework.png" width="90%"></div>

```
@inproceedings{gao2022decoupling,
  title={Decoupling Predictions in Distributed Learning for Multi-center Left Atrial MRI Segmentation},
  author={Gao, Zheyao and Li, Lei and Wu, Fuping and Wang, Sihan and Zhuang, Xiahai},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={517--527},
  year={2022},
  organization={Springer}
}
```

# Usage
```
python main.py --mode feddan --lq 0.7 --weight 0.01 --size 256
```

If you have any problems, please feel free to contact us. Thanks for your attention.
