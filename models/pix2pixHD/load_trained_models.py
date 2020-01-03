from keras.models import Model, save_model, load_model
from keras.optimizers import Adam
from .utils.conv2d_r import Conv2D_r
from keras.utils import multi_gpu_model
from .utils.instance_normalization import InstanceNormalization
import tensorflow as tf
from keras import backend as K
from .utils.sn import ConvSN2D, DenseSN


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)


class CoreGeneratorEnhancer():
    """Core Generator Enhancer.

    # 参数
        width: 图像宽度像素
        height: 图像高度像素
        channels: 输入图像和生成图像的通道数
        gpus: 使用的gpu数目 
    """

    def __init__(self,
                 weight_dir='./weights/',
                 gpus=0):

        self.gpus = gpus

        core_generator_original = load_model(weight_dir + 'core_generator.h5', custom_objects={
                                             'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})
        core_generator = Model(inputs=core_generator_original.input, outputs=[
                               core_generator_original.output, core_generator_original.get_layer('core_features_org').output])
        core_generator.name = "core_generator"
        core_generator.trainable = True

        self.model = core_generator
        self.save_model = core_generator


class CoreGenerator():
    """Core Generator.

    # 参数
        width: 图像宽度像素值
        height: 图像高度像素值
        channels: 输入图片和生成图片的通道数
        gpus: 使用的gpu数目 
    """

    def __init__(self,
                 weight_dir='./weights/',
                 gpus=0):

        self.gpus = gpus

        core_generator = load_model(weight_dir + 'core_generator.h5', custom_objects={
                                    'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})
        # core_generator = Model(inputs=core_generator_original.input,
        #                    outputs=[core_generator_original.get_layer('core_features_org').output, # core_generator_original.get_layer('core_features_true').output])
        core_generator.name = "core_generator"
        core_generator.trainable = True

        self.model = core_generator
        self.save_model = core_generator


class Enhancer():
    """Enhancer.

    # 参数
        width: 图像宽度像素值
        height: 图像高度像素值
        channels: 输入图片和生成图片的通道数
        gpus: 使用的gpu数目 
    """

    def __init__(self,
                 weight_dir='./weights/',
                 gpus=0):

        self.gpus = gpus

        enhancer = load_model(weight_dir + 'enhancer.h5', custom_objects={
                              'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})
        enhancer.name = 'enhancer'
        enhancer.trainable = True

        self.model = enhancer
        self.save_model = enhancer


class DiscriminatorFull():
    """Core Discriminator.

    # 参数
        width: 图像宽度像素值
        height: 图像高度像素值
        channels: 输入图片和生成图片的通道数
        gpus: 使用的gpu数目 
        learning_rate: 学习率
        decay_rate: 衰变率
    """

    def __init__(self,
                 weight_dir='./weights/',
                 learning_rate=0.0002,
                 decay_rate=2e-6,
                 gpus=1):

        self.gpus = gpus
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        def zero_loss(y_true, y_pred):
            return K.zeros_like(y_true)

        discriminator_full = load_model(weight_dir + 'discriminator_full.h5', custom_objects={
                                        'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'zero_loss': zero_loss, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})

        discriminator_full.trainable = True
        discriminator_full.name = "discriminator_full"

        self.model = discriminator_full
        self.save_model = discriminator_full


class DiscriminatorLow():
    """Low Discriminator.

    # 参数
        width: 图像宽度像素值
        height: 图像高度像素值
        channels: 输入图片和生成图片的通道数
        gpus: 使用的gpu数目 
        learning_rate: 学习率
        decay_rate: 衰变率
    """

    def __init__(self,
                 weight_dir='./weights/',
                 learning_rate=0.0002,
                 decay_rate=2e-6,
                 gpus=0):

        self.gpus = gpus
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        def zero_loss(y_true, y_pred):
            return K.zeros_like(y_true)

        discriminator_low = load_model(weight_dir + 'discriminator_low.h5', custom_objects={
                                       'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'zero_loss': zero_loss, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})

        discriminator_low.trainable = True
        discriminator_low.name = "discriminator_low"

        self.model = discriminator_low
        self.save_model = discriminator_low


class StyleFeatures():
    """Style Features.

    # 参数
        width: 图像宽度像素值
        height: 图像高度像素值
        channels: 输入图片和生成图片的通道数
        gpus: 使用的gpu数目 
        learning_rate: 学习率
        decay_rate: 衰变率
    """

    def __init__(self,
                 weight_dir='./weights/',
                 gpus=0):

        self.gpus = gpus

        style_features = load_model(weight_dir + 'style_features.h5', custom_objects={
                                    'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})

        style_features.trainable = True
        style_features.name = "style_features"

        self.model = style_features
        self.save_model = style_features
