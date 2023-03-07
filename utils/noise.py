import numpy as np
import random
from PIL import Image
# from pandas.plotting import radviz
class Uniform_noise(object):
    """增加高斯噪声
    此函数用将产生的均匀噪声加到图片上
    传入:
        img   :  原图
        low  :  均匀左界
        high :  均匀右界
        p: 概率
    返回:
        uniform_out : 噪声处理后的图片
    """

    def __init__(self, low, high, p):
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            # 将图片灰度标准化
            img_ = np.array(img).copy()
            d = len(img_.shape)
            img_ = img_ / 255.0
            # 产生高斯 noise
            noise = np.random.uniform(self.low, self.high, img_.shape)
            # 将噪声和图片叠加
            uniform_out = img_ + noise
            # 将超过 1 的置 1，低于 0 的置 0
            uniform_out = np.clip(uniform_out, 0, 1)
            # 将图片灰度范围的恢复为 0-255
            uniform_out = np.uint8(uniform_out * 255)
            # 将噪声范围搞为 0-255
            # noise = np.uint8(noise*255)
            if d == 2:
                return Image.fromarray(uniform_out).convert('1')
            elif d == 3:
                return Image.fromarray(uniform_out).convert('RGB')
        else:
            return img

#代码中的noisef为信号等级，例如我需要0.7的噪声，传入参数我传入的是1-0.7=0.3
class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))  # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


class Gaussian_noise(object):
    """增加高斯噪声
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
        p: 概率
    返回:
        gaussian_out : 噪声处理后的图片
    """

    def __init__(self, mean, sigma, p):
        self.mean = mean
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            # 将图片灰度标准化
            img_ = np.array(img).copy()
            img_ = img_ / 255.0
            # 产生高斯 noise
            noise = np.random.normal(self.mean, self.sigma, img_.shape)
            # 将噪声和图片叠加
            gaussian_out = img_ + noise
            # 将超过 1 的置 1，低于 0 的置 0
            gaussian_out = np.clip(gaussian_out, 0, 1)
            # 将图片灰度范围的恢复为 0-255
            gaussian_out = np.uint8(gaussian_out * 255)
            # 将噪声范围搞为 0-255
            # noise = np.uint8(noise*255)
            return Image.fromarray(gaussian_out).convert('RGB')
        else:
            return img