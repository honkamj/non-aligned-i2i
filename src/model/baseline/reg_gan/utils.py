"""Mostly copied from https://github.com/Kid-Liet/Reg-GAN"""


# Code copied from elsewhere
# pylint: disable-all


import numpy as np
import torch
import torch.nn.functional as F


class Resize:
    def __init__(self, size_tuple, use_cv=True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv

    def __call__(self, tensor):
        """
        Resized the tensor to the specific size
        Arg:    tensor  - The torch.Tensor obj whose rank is 4
        Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)

        tensor = F.interpolate(tensor, size=[self.size_tuple[0], self.size_tuple[1]])

        tensor = tensor.squeeze(0)

        return tensor  # 1, 64, 128, 128


class ToTensor:
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        return torch.from_numpy(tensor)


def tensor2image(tensor):
    image = (127.5 * (tensor.cpu().float().numpy())) + 127.5
    image1 = image[0]
    for i in range(1, tensor.shape[0]):
        image1 = np.hstack((image1, image[i]))

    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    # print ('image1.shape:',image1.shape)
    return image1.astype(np.uint8)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return d
