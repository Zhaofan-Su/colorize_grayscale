## 生成对抗网络应用：黑白图像上色

概述
---

随着深度学习的不断发展，新的网络形式——生成对抗网络（Generative Adversarial Network）GAN产生了，这是一种新的无监督式深度学习方法，由[伊恩·古德费洛](https://zh.wikipedia.org/wiki/%E4%BC%8A%E6%81%A9%C2%B7%E5%8F%A4%E5%BE%B7%E8%B4%B9%E6%B4%9B)等人于2014年提出。

卷积神经网络之父Yann Le Cun评价GAN网络是“机器学习这二十年来最酷的想法”。通过GAN网络，我们可以生成以假乱真的图片、影片以及三维物体模型等。GAN网络虽然是无监督学习网络，但是它对半监督学习、完全监督学习、强化学习都有很大的帮助。

本项目使用基于cGAN提出的pix2pix算法，完成对黑色图像自动上色的任务。


项目运行方式
---

使用以下命令运行项目:
```sh
cd colorize_grayscale

pip install keras tensorflow pillow h5py scikit-image

python colorize_base.py
```

项目提供了预先训练的[权重](https://drive.google.com/file/d/1Vpd-6CpF4pVzmkOPd7rqyYP1OOuZaRrd/view)，请将其放在`colorize_grayscale/weights/`文件夹下。这个权重基于一些当代人物的摄影图像进行训练，图像中人物不同，但是图像的数目不多。


