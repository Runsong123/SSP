import torch
import utils.general as utils
import abc
import numpy as np


class Sampler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_points(self, pc_input):
        pass

    @staticmethod
    def get_sampler(sampler_type):

        return utils.get_class("model.sample.{0}".format(sampler_type))


class NormalPerPoint(Sampler):
    def __init__(self, global_sigma, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input, local_sigma=None):
        batch_size, sample_size, dim = pc_input.shape

        if local_sigma is not None:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma.unsqueeze(-1))
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)

        sample_global = (torch.rand(batch_size, sample_size // 8, dim, device=pc_input.device) *
                         (self.global_sigma * 2)) - self.global_sigma

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample

    def get_points_empty(
        self,
        pc_input,
        grid_size,
        uniform_free_grid_num,
        uniform_free_sample_num_every_grid=10,
    ):
        sample_size, batch_size, dim = pc_input.shape

        scale = 2.0 / grid_size

        offset = torch.from_numpy(
            np.random.uniform(-1 * scale / 2, scale / 2 + 1e-9, uniform_free_grid_num * 10 * dim).reshape(uniform_free_grid_num, 10, dim))
        offset = torch.clamp(offset, -1 * scale / 2, scale / 2).cuda().float()
        # print(pc_input.shape)
        sample_local = pc_input[:, :, :] + offset
        sample_local = sample_local.reshape(uniform_free_sample_num_every_grid * uniform_free_grid_num, dim)

        return sample_local