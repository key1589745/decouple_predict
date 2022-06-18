# Decoupling Predictions in Distributed Learning for Multi-Center Left Atrial MRI Segmentation
This project is developed for our MICCAI 2022 paper: [Decoupling Predictions in Distributed Learning
for Multi-Center Left Atrial MRI Segmentation](https://arxiv.org/abs/2206.05284v1). Our code is implemented based on the [Learn_Noisy_Labels_Medical_Images](https://github.com/moucheng2017/Learn_Noisy_Labels_Medical_Images) and [probabilistic_unet](https://github.com/SimonKohl/probabilistic_unet), but we used them to tackle the non-IID challenge in distributed leearning. For more information, please read the following paper:

<div align=center><img src="framework.png" width="90%"></div>

```
@article{Gao2022decouple,
  title={Decoupling Predictions in Distributed Learning for Multi-Center Left Atrial MRI Segmentation},
  author={Zheyao, Gao and Zhuang, Xiahai},
  journal={arXiv preprint arXiv:2206.05284},
  year={2022}
}
```

# Datasets
```
XXX_dataset/
  -- TestSet/
      --images/
      --labels/
  -- train/
      --images/
      --labels/
  -- val/
      --images/
      --labels/
```

# Usage
```
python main.py --mode feddan --lq 0.7 --weight 0.01 --size 256
```

If you have any problems, please feel free to contact us. Thanks for your attention.
