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

        return utils.get_class("model.sample_SSN.{0}".format(sampler_type))


class NormalPerPoint(Sampler):
    def __init__(self, global_sigma, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    

    def get_points(
        self,
        pc_input,
        grid_size,
        uniform_free_grid_num,
        uniform_free_sample_num_every_grid=10,
    ):
        sample_size, batch_size, dim = pc_input.shape

        scale = 2.0/grid_size
        # pr(scale)

        off_set_matrix = np.zeros([uniform_free_sample_num_every_grid, 3])
        off_set_matrix[0, :] = np.array([0, 0, scale / 4])
        off_set_matrix[1, :] = np.array([0, 0, -1 * scale / 4])

        off_set_matrix[2, :] = np.array([0, 1 * scale / 4, 0])
        off_set_matrix[3, :] = np.array([0, -1 * scale / 4, 0])

        off_set_matrix[4, :] = np.array([1 * scale / 4, 0, 0])
        off_set_matrix[5, :] = np.array([-1 * scale / 4, 0, 0])

        # np.savetxt("offset.txt",off_set_matrix)

        off_set_matrix = torch.from_numpy(off_set_matrix).cuda().float()
        off_set_matrix = off_set_matrix.unsqueeze(0)

        random_noise = torch.cat(
            [
                torch.randn([uniform_free_grid_num, 6, dim]) * scale / 4,
                torch.randn([uniform_free_grid_num, uniform_free_sample_num_every_grid-6, dim]) * scale / 2,
            ],
            dim=1,
        )
        random_noise = torch.clamp(random_noise, -1 * scale / 2, scale / 2)
        

        # print(pc_input.shape)
        sample_local = pc_input[:, :, :] + (random_noise).cuda() + off_set_matrix
        sample_local = sample_local.reshape(uniform_free_sample_num_every_grid * uniform_free_grid_num, dim)

        
        # if local_sigma is not None:
        #     sample_local = pc_input + (torch.randn_like(pc_input) * scale)
        # else:
        #     sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)

        # sample_global = (torch.rand(batch_size, sample_size // 8, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma

        # sample = torch.cat([sample_local, sample_global], dim=1)

        return sample_local
