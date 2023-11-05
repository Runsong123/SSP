import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from pyhocon import ConfigFactory
import numpy as np
import argparse
import GPUtil
import torch
import utils.general as utils
from model.sample_SSN_uniform import Sampler
from model.network import gradient
from scipy.spatial import cKDTree
from utils.plots import plot_surface, plot_cuts, plot_cuts_axis, plot_threed_scatter
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as offline
from plotly.subplots import make_subplots
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_cuts_v1(
        decode_points_func,
        box_size=(1.0, 1.0, 1.0),
        max_n_eval_pts=1e6,
        resolution=256,
        thres=0.0,
        imgs_per_cut=1,
        save_path=None,
) -> go.Figure:
    """ plot levelset at a certain cross section, assume inputs are centered
    Args:
        decode_points_func: A function to extract the SDF/occupancy logits of (N, 3) points
        box_size (List[float]): bounding box dimension
        max_n_eval_pts (int): max number of points to evaluate in one inference
        resolution (int): cross section resolution xy
        thres (float): levelset value
        imgs_per_cut (int): number of images for each cut (plotted in rows)
    Returns:
        a numpy array for the image
    """
    xmax, ymax, zmax = [b / 2 for b in box_size]
    xx, yy = np.meshgrid(np.linspace(-xmax, xmax, resolution), np.linspace(-ymax, ymax, resolution))
    xx = xx.ravel()
    yy = yy.ravel()

    fig = make_subplots(
        rows=imgs_per_cut,
        cols=3,
        subplot_titles=("xz", "xy", "yz"),
        shared_xaxes="all",
        shared_yaxes="all",
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
    )

    def _plot_cut(fig, idx, pos, decode_points_func, xmax, ymax, resolution):
        """ plot one cross section pos (3, N) """
        # evaluate points in serial
        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()
        values = decode_points_func(field_input).flatten()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        values = values.reshape(resolution, resolution)
        contour_dict = dict(
            autocontour=False,
            colorscale="RdBu",
            contours=dict(
                start=-0.2 + thres,
                end=0.2 + thres,
                size=0.05,
                showlabels=True,  # show labels on contours
                labelfont=dict(
                    size=12,
                    color="white",
                ),  # label font properties
            ),
        )
        r_idx = idx // 3

        fig.add_trace(
            go.Contour(x=np.linspace(-xmax, xmax, resolution), y=np.linspace(-ymax, ymax, resolution), z=values, **contour_dict),
            col=idx % 3 + 1,
            row=r_idx + 1,  # 1-idx
        )

        fig.update_xaxes(
            range=[-xmax, xmax],  # sets the range of xaxis
            constrain="range",  # meanwhile compresses the xaxis by decreasing its "domain"
            col=idx % 3 + 1,
            row=r_idx + 1,
        )
        fig.update_yaxes(range=[-ymax, ymax], col=idx % 3 + 1, row=r_idx + 1)

    steps = np.stack([np.linspace(-b / 2, b / 2, imgs_per_cut + 2)[1:-1] for b in box_size], axis=-1)
    for index in range(imgs_per_cut):
        position_cut = [
            np.vstack([xx, np.full(xx.shape[0], steps[index, 1]), yy]),
            np.vstack([xx, yy, np.full(xx.shape[0], steps[index, 2])]),
            np.vstack([np.full(xx.shape[0], steps[index, 0]), xx, yy]),
        ]
        _plot_cut(fig, index * 3, position_cut[0], decode_points_func, xmax, zmax, resolution)
        _plot_cut(fig, index * 3 + 1, position_cut[1], decode_points_func, xmax, ymax, resolution)
        _plot_cut(fig, index * 3 + 2, position_cut[2], decode_points_func, ymax, zmax, resolution)

    fig.update_layout(
        title="iso-surface",
        height=512 * imgs_per_cut,
        width=512 * 3,
        autosize=False,
        scene=dict(aspectratio=dict(x=1, y=1)),
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)

    return fig


class ReconstructionRunner:
    def initial_loss_container(self):
        N_number = self.empty_data.shape[0]
        self.empty_loss_log = np.ones([N_number]) * 100
        self.uniform_loss_log = np.ones([self.surface_grid.shape[0]]) * 100
        self.uniform_normal_loss_log = np.ones([self.surface_grid.shape[0]]) * 100
        self.uniform_free_space_loss_log = np.ones([self.other_data.shape[0]]) * 100
        self.uniform_free_space_normal_loss_log = np.ones([self.other_data.shape[0]]) * 100

    def initial_sample_parameter(self):
        self.random_sample_num_every_grid = 50
        self.random_grid_num = self.points_batch // (9 * self.random_sample_num_every_grid)

        self.uniform_sample_num_every_grid = 50
        self.uniform_grid_num = self.points_batch // (3 * self.uniform_sample_num_every_grid)

        self.uniform_normal_sample_num_every_grid = 50
        self.uniform_normal_grid_num = self.points_batch // (3 * self.uniform_normal_sample_num_every_grid)

        self.uniform_free_sample_num_every_grid = 10
        self.uniform_free_grid_num = self.points_batch // (9 * self.uniform_free_sample_num_every_grid)

        self.uniform_normal_free_sample_num_every_grid = 10
        self.uniform_normal_free_grid_num = self.points_batch // (9 * self.uniform_normal_free_sample_num_every_grid)

        self.empty_points_num = self.points_batch // 10

    def sample_empty(self, epoch):
        N_number = self.empty_data.shape[0]
        # sample_range = min(
        #     self.points_batch * 10 + int(epoch * self.empty_data.shape[0] / self.nepochs * 10), self.empty_data.shape[0]
        # )

        # min_limit = 1e-5
        # if (self.empty_loss_log > min_limit).sum() == 0:
        #     self.empty_loss_log = np.ones([N_number]) * 100
        # elif (self.empty_loss_log > min_limit).sum() < self.empty_points_num:
        #     min_num = (self.empty_loss_log[self.empty_loss_log > min_limit]).min()
        #     self.empty_loss_log += min_num

        if self.empty_points_num >= N_number:
            empty_indices_repeat = True
        else:
            empty_indices_repeat = False

        empty_indices = (np.random.choice(
            np.arange(N_number),
            self.empty_points_num,
            True,
            p=self.empty_loss_log.ravel() / np.sum(self.empty_loss_log),
        ))
        empty_pts = self.empty_data[empty_indices]
        return empty_indices, empty_pts

    def random_grid_sample(self):

        if self.random_grid_num >= self.surface_grid.shape[0]:
            indices_repeat = True
        else:
            indices_repeat = False

        indices = np.random.choice(self.surface_grid.shape[0], self.random_grid_num, indices_repeat)
        random_gird = self.surface_grid[indices]
        random_ind = []

        # for p in np.array_split(random_gird, 10, axis=0):
        #     # random_noise = self.grid_scale * np.random.random_sample(p.shape) - self.grid_scale / 2
        #     d = self.ptree.query(p, self.random_sample_num_every_grid)
        #     random_ind.extend(d[1][:, :].reshape(-1))
        d = self.ptree.query(random_gird, self.random_sample_num_every_grid)
        random_ind = torch.from_numpy(d[1][:, :].reshape(-1))

        # random_ind = torch.from_numpy(np.asarray(random_ind).reshape(-1))
        # print()
        cur_data = self.data[random_ind]
        return indices, random_ind, random_gird, cur_data

    def sample_from_various_container(self):

        uniform_indice = np.random.choice(
            self.surface_grid.shape[0],
            self.uniform_grid_num,
            True,
            p=self.uniform_loss_log / np.sum(self.uniform_loss_log),
        )
        uniform_indice_normal = np.random.choice(
            self.surface_grid.shape[0],
            self.uniform_normal_grid_num,
            True,
            p=self.uniform_normal_loss_log / np.sum(self.uniform_normal_loss_log),
        )

        uniform_random_indice_freespace = np.random.choice(self.other_data.shape[0], self.uniform_free_grid_num, False)
        uniform_indice_freespace = np.random.choice(
            self.other_data.shape[0],
            self.uniform_free_grid_num,
            True,
            p=self.uniform_free_space_loss_log / np.sum(self.uniform_free_space_loss_log),
        )
        uniform_indice_freespace_normal = np.random.choice(
            self.other_data.shape[0],
            self.uniform_normal_free_grid_num,
            True,
            p=self.uniform_free_space_normal_loss_log / np.sum(self.uniform_free_space_normal_loss_log),
        )
        uniform_indice = np.concatenate([uniform_indice, uniform_indice_normal], axis=0)
        uniform_indice_free = np.concatenate([uniform_indice_freespace, uniform_indice_freespace_normal, uniform_random_indice_freespace],
                                             axis=0)
        # print(uniform_indice.shape)

        uniform_gird = self.surface_grid[uniform_indice]

        uniform_free_grid = self.other_data[uniform_indice_free, :]

        # uniform_ind = []
        # for p in np.array_split(uniform_gird[:self.uniform_grid_num, :], 10, axis=0):
        #     random_noise = self.grid_scale * np.random.random_sample(p.shape) - self.grid_scale / 2
        #     d = self.ptree.query(p + random_noise, self.uniform_sample_num_every_grid)
        #     uniform_ind.extend(d[1][:, :].reshape(-1))
        # for p in np.array_split(uniform_gird[self.uniform_grid_num:, :], 10, axis=0):
        #     random_noise = self.grid_scale * np.random.random_sample(p.shape) - self.grid_scale / 2
        #     d = self.ptree.query(p + random_noise, self.uniform_normal_sample_num_every_grid)
        #     uniform_ind.extend(d[1][:, :].reshape(-1))
        random_noise = self.grid_scale * np.random.random_sample(uniform_gird.shape) - self.grid_scale / 2
        d = self.ptree.query(uniform_gird + random_noise, self.uniform_normal_sample_num_every_grid)
        uniform_ind = d[1][:, :].reshape(-1)
        # print(np.asarray(uniform_ind).reshape(-1))
        uniform_ind = torch.from_numpy(np.asarray(uniform_ind).reshape(-1))
        # print()
        uniform_data = self.data[uniform_ind]

        return uniform_indice, uniform_indice_free, uniform_gird, uniform_free_grid, uniform_ind, uniform_data

    def losslog_save(self, epoch):
        utils.mkdir_ifnotexists("%s/%s/" % (self.checkpoints_path, "save_midden"))
        np.savetxt(
            "%s/%s/empty_%d.txt" % (self.checkpoints_path, "save_midden", epoch),
            np.concatenate(
                [
                    self.empty_data,
                    np.expand_dims(self.empty_loss_log, axis=1),
                ],
                axis=1,
            ),
        )
        np.savetxt(
            "%s/%s/surface_%d.txt" % (self.checkpoints_path, "save_midden", epoch),
            np.concatenate(
                [self.surface_grid, np.expand_dims(self.uniform_loss_log, axis=1)],
                axis=1,
            ),
        )
        np.savetxt(
            "%s/%s/surface_normal_%d.txt" % (self.checkpoints_path, "save_midden", epoch),
            np.concatenate(
                [
                    self.surface_grid,
                    np.expand_dims(self.uniform_normal_loss_log, axis=1),
                ],
                axis=1,
            ),
        )
        np.savetxt(
            "%s/%s/surface_freespace_%d.txt" % (self.checkpoints_path, "save_midden", epoch),
            np.concatenate(
                [
                    self.other_data,
                    np.expand_dims(self.uniform_free_space_loss_log, axis=1),
                ],
                axis=1,
            ),
        )
        np.savetxt(
            "%s/%s/surface_freespace_normal_%d.txt" % (self.checkpoints_path, "save_midden", epoch),
            np.concatenate(
                [
                    self.other_data,
                    np.expand_dims(self.uniform_free_space_normal_loss_log, axis=1),
                ],
                axis=1,
            ),
        )

    def calculate_frequency_density(self, nonmnfld_pnts):
        nonmnfld_pnts_numpy = nonmnfld_pnts.detach().cpu().numpy()
        nonmnfld_near_all_ind = []
        nearest_ind = []
        top1_neighbor_num = 4
        unit_point_number = []

        dis_all = []
        for p in np.array_split(nonmnfld_pnts_numpy, 100, axis=0):
            #  self.median_dis default for 25

            # near_num = query_set[0]
            dis_item = self.ptree.query(p, 1)
            dis_all.append(dis_item[0])
            nearest_ind.append(dis_item[1])

        # nonmnfld_near_ind = np.concatenate(frequency_ind, axis=0)[:, 1:].reshape(-1)
        nonmnfld_nearest_ind = np.concatenate(nearest_ind, axis=0)
        dis_numpy = np.concatenate(dis_all).reshape(-1)
        non_surface_pred_GT = torch.from_numpy(dis_numpy).float().cuda()
        # non_surface_pred_GT = torch.from_numpy(np.concatenate(dis_all).reshape(-1)).float().cuda()
        # print(nonmnfld_unit_point_number)

        nonmnfld_normal_point = self.data[nonmnfld_nearest_ind, :3].reshape(-1, 3)
        # nonmnfld_corresponding_pts_GT = self.data[nonmnfld_nearest_ind, :3]
        nonmnfld_normal_GT = self.data[nonmnfld_nearest_ind, 3:].reshape(-1, 1, 3)
        # nonmnfld_near_pts_normal = self.data[nonmnfld_near_ind, 3:].reshape(-1, 3, 3)

        # diff_vector = nonmnfld_corresponding_pts_GT - nonmnfld_pnts
        # non_surface_pred_GT = torch.norm(diff_vector, dim=1)

        # print(diff_vector.shape)
        # print(nonmnfld_near_unit_dis.shape)
        # distance = (torch.norm(diff_vector, dim=1)**2 - torch.sum(diff_vector*self.data[nonmnfld_nearest_ind,3:], dim=1)**2 ) /(nonmnfld_near_unit_dis**2 + 1e-6)
        # print(torch.isnan(distance).sum())
        # print(distance)
        # similarity = F.cosine_similarity(nonmnfld_normal_GT, nonmnfld_near_pts_normal, dim=2)
        # # similarity_weight = similarity.abs().mean(dim=1)
        # frequency_weight = similarity.abs().mean(dim=1)

        # density_confidence = 1 - 1.0 / (1 + torch.exp(-100 * ((1 - nonmnfld_unit_point_number / 25.0) - 0.5)))

        return nonmnfld_normal_point, non_surface_pred_GT, nonmnfld_normal_GT, dis_numpy

    def loss_for_empty_value(self, outside_loss, empty_indices):
        outside_loss_numpy = (outside_loss).detach().cpu().numpy().reshape(-1, 10)
        outside_loss_numpy = np.mean(outside_loss_numpy, axis=1)
        # print(uniform_indice_normal_near_loss.shape)
        self.empty_loss_log[empty_indices] = (self.empty_loss_log[empty_indices] * 0.1 + (outside_loss_numpy + 1e-5) * 0.9)

    def loss_for_no_surface_value(self, non_surface_loss, uniform_indice_free):
        # print(n)
        uniform_free_space_loss = (non_surface_loss).detach().cpu().numpy().reshape(-1, 10)
        uniform_free_space_loss = np.mean(uniform_free_space_loss, axis=1)
        # print(uniform_free_space_loss.shape)
        # print(uniform_indice_free.shape)
        self.uniform_free_space_loss_log[uniform_indice_free] = (self.uniform_free_space_loss_log[uniform_indice_free] * 0.1 +
                                                                 uniform_free_space_loss * 0.9)

    def loss_for_surface_value(self, mnfld_loss, indices, uniform_indice):
        surface_grid_loss = mnfld_loss.detach().cpu().numpy().reshape(-1, self.random_sample_num_every_grid)
        surface_grid_loss = np.mean(surface_grid_loss, axis=1)

        # non_mnfld_loss_surface_part = non_mnfld_loss.detach().cpu().numpy()[: hybird_grid.shape[0] * 10].reshape(-1, 10)
        # non_mnfld_loss_surface_part = np.mean(non_mnfld_loss_surface_part, axis=1)

        surface_grid_loss_all = surface_grid_loss
        self.uniform_loss_log[np.concatenate([indices, uniform_indice],
                                             axis=0)] = (self.uniform_loss_log[np.concatenate([indices, uniform_indice], axis=0)] * 0.1 +
                                                         surface_grid_loss_all * 0.9)

        # return surface_grid_loss

    def loss_for_no_surface_normal(self, non_surface_normal_loss, uniform_indice_free):
        # print(non_surface_normal_loss.shape)
        uniform_free_space_normal_loss = (non_surface_normal_loss.detach().cpu().numpy().reshape(-1, 10))
        uniform_free_space_normal_loss = np.mean(uniform_free_space_normal_loss, axis=1)
        # print(uniform_indice_normal_near_loss.shape)
        self.uniform_free_space_normal_loss_log[uniform_indice_free] = (self.uniform_free_space_normal_loss_log[uniform_indice_free] * 0.1 +
                                                                        uniform_free_space_normal_loss * 0.9)

    def loss_for_surface_normal(self, normals_loss, indices, uniform_indice):
        surface_grid_normal_loss = normals_loss.detach().cpu().numpy().reshape(-1, self.random_sample_num_every_grid)
        surface_grid_normal_loss = np.mean(surface_grid_normal_loss, axis=1)

        surface_grid_normal_loss_all = surface_grid_normal_loss
        self.uniform_normal_loss_log[np.concatenate(
            [indices, uniform_indice],
            axis=0)] = (self.uniform_loss_log[np.concatenate([indices, uniform_indice], axis=0)] * 0.1 + surface_grid_normal_loss_all * 0.9)

    def get_scale(self, epoch):

        loss_scale = 1
        if epoch < 2000:
            loss_scale = 1
        elif epoch >= 2000 and epoch < 5000:
            loss_scale = 10
        elif epoch >= 5000:
            loss_scale = 30
        return loss_scale * self.size

    def loss_scale_weight(self, GT, pred, scale):
        GT_weight = torch.exp(-scale * GT) + 0.01
        pred_weight = torch.exp(-scale * torch.abs(pred)) + 0.01
        # print(pred.shape)
        # print(GT.shape)
        # print(torch.max(GT_weight,pred_weight).shape)
        return torch.max(GT_weight, pred_weight)

    def update_ratio(self, epoch):
        value_ratio = 40
        if epoch < 10000:
            value_ratio = 40
        else:
            value_ratio = max(40 - (epoch - 10000) // 1000, 20)

        return value_ratio

    # def update_ratio(self, epoch):
    #     value_ratio = 40
    #     if epoch < 10000:
    #         value_ratio = 40
    #     else:
    #         value_ratio = max(40 - (epoch - 10000) // 2000 * 10, 1)

    #     return value_ratio

    def run(self):

        print("running")

        self.data = self.data.cuda()
        self.data.requires_grad_()
        ####################empty
        # self.empty_data = self.empty_data.cuda()
        # self.empty_data.requires_grad_()

        if self.eval:

            print("evaluating epoch: {0}".format(self.startepoch))
            my_path = os.path.join(self.cur_exp_dir, "evaluation", str(self.startepoch))

            utils.mkdir_ifnotexists(os.path.join(self.cur_exp_dir, "evaluation"))
            utils.mkdir_ifnotexists(my_path)

            self.plot_shapes(epoch=self.startepoch, path=my_path, with_cuts=False)
            return

        print("training")

        ######## initail sample parameter
        self.initial_loss_container()
        self.initial_sample_parameter()

        # uniform_near_surface_loss_log = np.ones([self.surface_grid.shape[0]]) * 100
        # uniform_near_surface_normal_loss_log = np.ones([self.surface_grid.shape[0]]) * 100

        point_ratio = 1
        isDebug = True
        check_frequency = self.conf.get_int("train.checkpoint_frequency")
        if not isDebug:
            check_frequency = check_frequency * 5
        ######################## to here

        load_sample_time = 0.0
        calculated_offsurface_time = 0.0
        # load_surface_time =0.0

        for epoch in range(self.startepoch, self.nepochs + 1):
            # if epoch % 1000 == 0:
            #     self.network.set_mask(3 + epoch // 1000)
            # current_value_ratio = self.update_ratio(epoch)
            # if self.dataset[-5:]=="noise":
            #     current_value_ratio = 20
            # else:
            #     current_value_ratio = 40

            if epoch % (check_frequency) == 0:
                self.losslog_save(epoch)
                print("saving checkpoint: ", epoch)
                self.save_checkpoints(epoch)
                print("plot validation epoch: ", epoch)
                self.plot_shapes(epoch)

            current_value_ratio = 40

            load_sample_start = time.time()
            empty_indices, empty_grid_pts = self.sample_empty(epoch)
            empty_grid_pts_pns = torch.from_numpy(empty_grid_pts).float().cuda().requires_grad_()
            empty_pnts = self.sampler.get_points(empty_grid_pts_pns.unsqueeze(1), self.size, self.empty_points_num, 10).squeeze()

            indices, random_ind, random_gird, cur_data = self.random_grid_sample()

            (
                uniform_indice,
                uniform_indice_free,
                uniform_gird,
                uniform_free_grid,
                uniform_ind,
                uniform_data,
            ) = self.sample_from_various_container()

            cur_data = torch.cat([cur_data, uniform_data], dim=0)

            mnfld_pnts = cur_data[:, :self.d_in]
            # all_indice = torch.cat([indices,uniform_ind],dim=0)
            # mnfld_sigma = self.local_sigma[all_indice]

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)

            N_sample_point_cloud = mnfld_pnts.shape[0]

            load_sample_time += time.time() - load_sample_start

            # size = len()

            # near_nonmnfld_pnts = self.sampler.get_points_2(
            #     hybird_grid.unsqueeze(1), self.size, hybird_grid.shape[0], self.uniform_free_sample_num_every_grid
            # ).squeeze()
            # print(uniform_free_grid_pns.shape)
            # print((self.uniform_free_grid_num + self.uniform_normal_free_grid_num))
            # nonmnfld_pnts = self.sampler.get_points(
            #     uniform_free_grid_pns.unsqueeze(1),
            #     self.size,
            #     (self.uniform_free_grid_num * 3),
            #     self.uniform_free_sample_num_every_grid,
            # ).squeeze()
            # print(nonmnfld_pnts.shape)

            # np.savetxt("nonmnfld_pnts.txt", nonmnfld_pnts.detach().cpu().numpy())
            # np.savetxt("uniform_free_grid_pns.txt", uniform_free_grid)
            # exit()
            # print(nonmnfld_pnts.shape)

            # nonmnfld_pnts = torch.cat([near_nonmnfld_pnts, nonmnfld_pnts], dim=0)

            start_off_time = time.time()
            size = uniform_indice_free.shape[0]
            # N_freespace_point = nonmnfld_pnts.shape[0]
            random_choice = np.random.randint(1000, size=(size, 10)) + np.arange(size).reshape(-1, 1) * 1000

            nonmnfld_pnts_info = self.cal_data[uniform_indice_free].reshape(-1, 7)[random_choice.reshape(-1), :].reshape(-1, 10, 7)
            nonmnfld_pnts, non_surface_pred_GT, nonmnfld_normal_GT = nonmnfld_pnts_info[:, :, :
                                                                                        3], nonmnfld_pnts_info[:, :,
                                                                                                               3], nonmnfld_pnts_info[:, :,
                                                                                                                                      4:]
            # nonmnfld_normal_GT
            nonmnfld_pnts = torch.from_numpy(nonmnfld_pnts.reshape(-1, 3)).float().cuda().requires_grad_()
            non_surface_pred_GT = torch.from_numpy(non_surface_pred_GT.reshape(-1)).float().cuda().requires_grad_()
            nonmnfld_normal_GT = torch.from_numpy(nonmnfld_normal_GT.reshape(-1, 3)).float().cuda().requires_grad_()
            # print(nonmnfld_pnts.shape)
            # print(non_surface_pred_GT.shape)
            # print(uniform_indice_free.shape)
            # (nonmnfld_normal_point, non_surface_pred_GT, nonmnfld_normal_GT, dis_numpy) = self.calculate_frequency_density(nonmnfld_pnts)
            # weight = torch.from_numpy(np.exp(-1 * (dis_numpy)/ self.grid_scale)).float().cuda()
            # print(density_confidence)
            calculated_offsurface_time += time.time() - start_off_time
            # forward pass

            mnfld_pred = self.network(mnfld_pnts)
            nonmnfld_pred = self.network(nonmnfld_pnts)

            # nonmnfld_normal_point_pred = self.network(nonmnfld_normal_point)

            outside_pred = self.network(empty_pnts)

            # compute grad

            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

            # nonmnfld_normal_point_gradient = gradient(nonmnfld_normal_point, nonmnfld_normal_point_pred)

            outside_grad = gradient(empty_pnts, outside_pred)

            # manifold loss

            # non_surface_pred_GT = non_surface_pred_GT
            # print(non_surface_pred_GT.shape)
            # print(nonmnfld_pred.shape)
            # weight_scale = self.get_scale(epoch)
            # loss_weight = self.loss_scale_weight(non_surface_pred_GT, nonmnfld_pred.squeeze(1).detach(), weight_scale)
            non_mnfld_loss = torch.min(
                ((nonmnfld_pred.squeeze(1) - non_surface_pred_GT).abs()),
                ((nonmnfld_pred.squeeze(1) + non_surface_pred_GT).abs()),
            )
            non_mnfld_loss = non_mnfld_loss

            non_surface_loss = (non_mnfld_loss)
            self.loss_for_no_surface_value(non_mnfld_loss, uniform_indice_free)

            mnfld_loss = mnfld_pred.abs()
            # self.loss_for_surface_value(mnfld_loss, indices, uniform_indice)
            if epoch >= 2000:
                self.loss_for_surface_value(mnfld_loss, indices, uniform_indice)

            mnfld_loss = mnfld_loss.mean()
            # eikonal loss

            grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1)**2).mean() + ((outside_grad.norm(2, dim=-1) - 1)**2).mean()
            # grad_loss
            # grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

            loss_outside = torch.abs(torch.max(-1 * outside_pred + 1.0 / self.size, torch.full_like(outside_pred, 0)))
            # print(loss_outside.shape)
            self.loss_for_empty_value(loss_outside, empty_indices)

            loss_outside = loss_outside.mean() * 10

            if epoch >= 2000:
                loss = mnfld_loss * current_value_ratio + self.grad_lambda * grad_loss * (21 - (current_value_ratio + 1) // 2)
            else:
                loss = self.grad_lambda * grad_loss * (21 - (current_value_ratio + 1) // 2)
            # loss = mnfld_loss  + self.grad_lambda * grad_loss

            # normals loss

            if self.with_normals:
                normals = cur_data[:, -self.d_in:]

                ################ off-surface point value

                nonmnfld_normal_GT = nonmnfld_normal_GT.squeeze(1)
                # non_surface_normal_loss = torch.min(
                #     ((nonmnfld_grad - nonmnfld_normal_GT).abs()).norm(2, dim=1),
                #     ((nonmnfld_grad + nonmnfld_normal_GT).abs()).norm(2, dim=1),
                # )
                # non_surface_normal_loss = torch.min(
                #     ((nonmnfld_grad - nonmnfld_normal_GT).abs()).norm(2, dim=1),
                #     ((nonmnfld_grad + nonmnfld_normal_GT).abs()).norm(2, dim=1),
                # )

                non_surface_normal_loss_1 = torch.min(
                    ((nonmnfld_grad - nonmnfld_normal_GT).abs()).norm(2, dim=1),
                    ((nonmnfld_grad + nonmnfld_normal_GT).abs()).norm(2, dim=1),
                )

                # non_surface_normal_loss_1 = torch.abs(torch.max(non_surface_normal_loss_1 - 0.3, torch.full_like(non_surface_normal_loss_1, 0)))
                # non_surface_normal_loss_2 = torch.abs(torch.max(non_surface_normal_loss_2 - 0.1, torch.full_like(non_surface_normal_loss_2, 0)))

                # if epoch>=4000:
                # non_surface_normal_loss_2 = (nonmnfld_grad - nonmnfld_normal_point_gradient).norm(2,dim=1)
                # non_surface_normal_loss = (non_surface_normal_loss_1 + non_surface_normal_loss_2)/2
                # else:
                non_surface_normal_loss = non_surface_normal_loss_1
                # non_surface_normal_loss = non_surface_normal_loss
                # print(non_surface_normal_loss.shape)
                self.loss_for_no_surface_normal(non_surface_normal_loss, uniform_indice_free)
                non_surface_all_loss = (non_surface_normal_loss * (21 - (current_value_ratio + 1) // 2) + non_surface_loss *
                                        ((current_value_ratio + 1) // 2)).mean()

                # self.loss_for_no_surface_normal(hybird_grid, non_surface_normal_loss, uniform_indice_free)
                # non_surface_all_loss = (
                #     non_surface_normal_loss + non_surface_loss
                # ).mean()

                ################

                normals_loss = torch.min(((mnfld_grad - normals).abs()).norm(2, dim=1),
                                         ((mnfld_grad + normals).abs()).norm(2, dim=1))  # .mean()
                # self.loss_for_surface_normal()
                if epoch >= 2000:
                    self.loss_for_surface_normal(normals_loss, indices, uniform_indice)
                ########################################

                ########################################

                normals_loss = normals_loss.mean()
                if epoch >= 2000:
                    loss = loss + self.normals_lambda * normals_loss * (41 - current_value_ratio) + 0.5 * non_surface_all_loss
                else:
                    loss = loss + 0.5 * non_surface_all_loss
            else:
                normals_loss = torch.zeros(1)

            # back propagation

            loss = loss + loss_outside

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            self.writer.add_scalar("training/loss", loss.detach().item(), epoch)
            self.writer.add_scalar("training/loss_outside", loss_outside.detach().item(), epoch)
            self.writer.add_scalar("training/non_surface_loss", non_surface_loss.detach().mean().item(), epoch)
            self.writer.add_scalar("training/non_surface_normal_loss", non_surface_normal_loss.detach().mean().item(), epoch)
            self.writer.add_scalar("training/mnfld_loss", mnfld_loss.detach().mean().item(), epoch)
            self.writer.add_scalar("training/mnfld_normal_loss", (self.normals_lambda * normals_loss).detach().mean().item(), epoch)
            self.writer.add_scalar("training/grad_loss", (self.grad_lambda * grad_loss).detach().item(), epoch)

            if epoch % self.conf.get_int("train.status_frequency") == 0:
                print("load_sample_time", load_sample_time)
                print("calculated_offsurface_time", calculated_offsurface_time)
                print("Train Epoch: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tManifold loss: {:.6f}\tnon_surface_normal_loss:{:.6f}"
                      "\tGrad loss: {:.6f}\tNormals Loss: {:.6f}\tOutside Loss: {:.6f}".format(
                          epoch,
                          self.nepochs,
                          100.0 * epoch / self.nepochs,
                          loss.item(),
                          mnfld_loss.item(),
                          non_surface_all_loss.item(),
                          grad_loss.item(),
                          normals_loss.item(),
                          loss_outside.item(),
                      ))

    def plot_shapes(self, epoch, path=None, with_cuts=False):
        # plot network validation shapes
        with torch.no_grad():

            self.network.eval()

            if not path:
                path = self.plots_dir

            if self.points_batch >= self.data.shape[0]:
                indices_repeat = True
            else:
                indices_repeat = True
            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, indices_repeat))

            pnts = self.data[indices, :3]

            object_bounding_sphere = np.linalg.norm(pnts.cpu().numpy(), axis=1).max()

            box_size = (object_bounding_sphere * 2 + 0.2, ) * 3

            # plot_cuts_v1(lambda x: self.network(x).squeeze().detach(), box_size= box_size, max_n_eval_pts=10000, thres=0.0,save_path=path+"aaa.html")

            # plot_threed_scatter(pnts.cpu().numpy(),path,epoch,epoch)

            plot_surface(with_points=True,
                         points=pnts,
                         decoder=self.network,
                         path=path,
                         epoch=epoch,
                         shapename=self.expname,
                         **self.conf.get_config("plot"))

            if with_cuts:
                for i in range(3):
                    plot_cuts_axis(points=pnts, decoder=self.network, path=path, epoch=epoch, near_zero=False, axis=i)

    def __init__(self, **kwargs):

        self.home_dir = kwargs["home_dir"]  # "/apdcephfs/share_1467498/datasets/runsongzhu/IGR/things_emtpy_weight_10_v2"
        # utils.mkdir_ifnotexists(self.home_dir)
        # self.home_dir = os.path.abspath(os.pardir)

        self.file_dict = dict()
        # self.file_dict["thingi"] = "/thingi_normalized_data_with_PCA_normal/"
        self.file_dict["scene"] = "/optim_data/scene/"
        self.file_dict["deep_geometric_prior_data"] = "/optim_data/deep_geometric_prior_data_new/"
        self.file_dict["dfaust"] = "/optim_data/dfaust/"
        self.file_dict["vardensity_gradient"] = "/optim_data/pcpnet_adaFit/vardensity_gradient/"
        self.file_dict["vardensity_striped"] = "/optim_data/pcpnet_adaFit/vardensity_striped/"
        self.file_dict["abc_noisefree"] = "/optim_data/abc_adaFit/abc_noisefree/"
        self.file_dict["famous_noisefree"] = "/optim_data/famous_noisefree_adaFit/famous_noisefree/"

        self.file_dict["low_noise"] = "/optim_data/pcpnet_adaFit/low_noise/"
        self.file_dict["med_noise"] = "/optim_data/pcpnet_adaFit/med_noise/"
        self.file_dict["high_noise"] = "/optim_data/pcpnet_adaFit/high_noise/"
        self.file_dict["high_noise"] = "/optim_data/pcpnet_adaFit/high_noise/"
        self.file_dict["thingi"] = "/optim_data/thingi_normalized_concate_adaFit/thingi/"
        # config setting

        if type(kwargs["conf"]) == str:
            self.conf_filename = "./reconstruction/" + kwargs["conf"]
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs["conf"]

        self.expname = kwargs["expname"]

        # GPU settings

        self.GPU_INDEX = kwargs["gpu_index"]

        if not self.GPU_INDEX == "ignore":
            os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(self.GPU_INDEX)

        self.num_of_gpus = torch.cuda.device_count()

        self.eval = kwargs["eval"]

        self.dataset = kwargs["dataset"]

        # settings for loading an existing experiment
        self.home_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/exps/%s" % self.home_dir
        utils.mkdir_ifnotexists(self.home_dir)
        if (kwargs["is_continue"] or self.eval) and kwargs["timestamp"] == "latest":
            if os.path.exists(os.path.join(self.home_dir, "exps", self.expname)):
                timestamps = os.listdir(os.path.join(self.home_dir, "exps", self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs["timestamp"]
            is_continue = kwargs["is_continue"] or self.eval

        self.exps_folder_name = "exps"
        utils.mkdir_ifnotexists(os.path.join(self.home_dir, self.exps_folder_name))

        self.input_file = (self.conf.get_string("train.input_path") + self.file_dict[kwargs["dataset"]] + kwargs["shape"] + ".npy")
        print(self.input_file)
        import glob

        if kwargs["dataset"] == "abc_noisefree" or kwargs["dataset"] == "famous_noisefree" or kwargs["dataset"] == "thingi":

            files_name = glob.glob("%s_%s_adaptive_more_grid11_density_boundary_depth/%s_empty_*_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
            ))
            print(files_name)
            kwargs["size"] = int(files_name[0].split("_")[-2])
            self.size = kwargs["size"]
            self.grid_scale = 2.0 / self.size
            print(kwargs["size"])
            self.empty_file = "%s_%s_adaptive_more_grid11_density_boundary_depth/%s_empty_%d_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
                kwargs["size"],
            )
            self.surface_grid_file = "%s_%s_adaptive_more_grid11_density_boundary_depth/%s_grid_%d_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
                kwargs["size"],
            )
            self.other_grid_file = "%s_%s_adaptive_more_grid11_density_boundary_depth/%s_other_%d_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
                kwargs["size"],
            )
        elif kwargs["dataset"][-5:] == "noise":
            files_name = glob.glob("%s_%s_adaptive_more_grid11_max_100_density_boundary_depth_name_change/%s_empty_*_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
            ))
            print(files_name)
            kwargs["size"] = int(files_name[0].split("_")[-2])
            self.size = kwargs["size"]
            self.grid_scale = 2.0 / self.size
            print(kwargs["size"])
            self.empty_file = "%s_%s_adaptive_more_grid11_max_100_density_boundary_depth_name_change/%s_empty_%d_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
                kwargs["size"],
            )
            self.surface_grid_file = "%s_%s_adaptive_more_grid11_max_100_density_boundary_depth_name_change/%s_grid_%d_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
                kwargs["size"],
            )
            self.other_grid_file = "%s_%s_adaptive_more_grid11_max_100_density_boundary_depth_name_change/%s_other_%d_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
                kwargs["size"],
            )
        else:
            files_name = glob.glob("%s_%s_adaptive_more_grid11_max_100_density_boundary_depth/%s_empty_*_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
            ))
            print(files_name)
            kwargs["size"] = int(files_name[0].split("_")[-2])
            self.size = kwargs["size"]
            self.grid_scale = 2.0 / self.size
            print(kwargs["size"])
            self.empty_file = "%s_%s_adaptive_more_grid11_max_100_density_boundary_depth/%s_empty_%d_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
                kwargs["size"],
            )
            self.surface_grid_file = "%s_%s_adaptive_more_grid11_max_100_density_boundary_depth/%s_grid_%d_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
                kwargs["size"],
            )
            self.other_grid_file = "%s_%s_adaptive_more_grid11_max_100_density_boundary_depth/%s_other_%d_pts.txt" % (
                "/research/dept6/khhui/SSN_Fitting/data/data/adaptive_more_grid/output_res_1",
                kwargs["dataset"],
                kwargs["shape"],
                kwargs["size"],
            )

        self.data = utils.load_point_cloud_by_file_extension(self.input_file)
        self.empty_data = np.loadtxt(self.empty_file)
        self.other_data = np.loadtxt(self.other_grid_file)[:, :3]
        self.surface_grid = np.loadtxt(self.surface_grid_file)
        self.other_data = np.concatenate([self.other_data, self.surface_grid], axis=0)
        self.shape = kwargs["shape"]

        sigma_set = []
        ptree = cKDTree(self.data[:, :3])
        self.numpy_data = self.data[:, :3].numpy()
        # np.savetxt("data.txt",self.numpy_data)
        # exit()
        near_20_dis = []
        self.ptree = ptree
        for p in np.array_split(self.data[:, :3], 100, axis=0):
            d = ptree.query(p, 50 + 1)
            sigma_set.append(d[0][:, -1])
            near_20_dis.append(d[0][:, -25])
        self.median_dis = np.median(np.concatenate(near_20_dis))

        cache_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/data_prepare/data_precompute"
        # cache_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/reconstruction/data_precompute"
        cache_file = "%s/%s.npy" % (cache_dir, self.shape)
        if os.path.exists(cache_file):
            self.cal_data = np.load(cache_file)
        else:
            start_time = time.time()
            self.cal_data = self.precalc(self.other_data)
            np.save(cache_file, self.cal_data)
            print("consuming time: ", time.time() - start_time)

        # print(self.cal_data.shape)
        # exit()

        sigmas = np.concatenate(sigma_set)
        self.local_sigma = torch.from_numpy(sigmas).float().cuda()

        self.expdir = os.path.join(self.home_dir, self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)

        if is_continue:
            self.timestamp = timestamp
        else:
            self.timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())

        self.cur_exp_dir = os.path.join(self.expdir, self.timestamp)
        utils.mkdir_ifnotexists(self.cur_exp_dir)

        self.plots_dir = os.path.join(self.cur_exp_dir, "plots")
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.cur_exp_dir, "checkpoints")
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.checkpoints_path = os.path.join(self.cur_exp_dir, "checkpoints")
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        self.writer = SummaryWriter(self.checkpoints_path + "/log/")

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        self.nepochs = kwargs["nepochs"]

        self.points_batch = kwargs["points_batch"]

        self.global_sigma = self.conf.get_float("network.sampler.properties.global_sigma")
        self.sampler = Sampler.get_sampler(self.conf.get_string("network.sampler.sampler_type"))(self.global_sigma, self.local_sigma)
        self.grad_lambda = self.conf.get_float("network.loss.lambda")
        self.normals_lambda = self.conf.get_float("network.loss.normals_lambda")

        # use normals if data has  normals and normals_lambda is positive
        self.with_normals = self.normals_lambda > 0 and self.data.shape[-1] >= 6

        self.d_in = self.conf.get_int("train.d_in")

        self.network = utils.get_class(self.conf.get_string("train.network_class"))(d_in=self.d_in,
                                                                                    **self.conf.get_config("network.inputs"))

        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list("train.learning_rate_schedule"))
        self.weight_decay = self.conf.get_float("train.weight_decay")

        self.startepoch = 0

        self.optimizer = torch.optim.Adam([
            {
                "params": self.network.parameters(),
                "lr": self.lr_schedules[0].get_learning_rate(0),
                "weight_decay": self.weight_decay,
            },
        ])

        # if continue load checkpoints

        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, "checkpoints")

            saved_model_state = torch.load(os.path.join(old_checkpnts_dir, "ModelParameters", str(kwargs["checkpoint"]) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(os.path.join(old_checkpnts_dir, "OptimizerParameters", str(kwargs["checkpoint"]) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.startepoch = saved_model_state["epoch"]

    def precalc(self, other_data):
        scale = self.grid_scale
        grid_num = other_data.shape[0]
        print(grid_num)
        # calc_pts = np.zeros([grid_num, 8000, 4])
        calc_pts = []
        offset = np.mgrid[-1:1:0.2, -1:1:0.2, -1:1:0.2].reshape(3, -1).T
        offset = offset * scale / 2
        corres_normal = []
        # dis =
        # for i in range()
        i = 0
        for p in np.array_split(other_data, 5, axis=0):
            pts = np.expand_dims(p, axis=1) + offset.reshape(1, 1000, 3)
            pts = pts.reshape(-1, 3)
            d = self.ptree.query(pts, 1, workers=-1)
            index = d[1][:].reshape(-1)
            calc_pts.append(d[0][:].reshape(-1, 1000))
            corres_normal.append(self.data[index, 3:].reshape(-1, 1000, 3))
            # print(i)
            # i += 1
        corres_normal = np.concatenate(corres_normal, axis=0).reshape(-1, 1000, 3)
        calc_pts = np.concatenate(calc_pts, axis=0).reshape(-1, 1000, 1)
        offset_pts = np.expand_dims(other_data, axis=1) + offset.reshape(-1, 1000, 3)
        return np.concatenate([offset_pts, calc_pts, corres_normal], axis=2)

        # return calc_pts

    def get_learning_rate_schedules(self, schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    ))

            else:
                raise Exception('no known learning rate schedule of type "{}"'.format(schedule_specs["Type"]))

        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self, epoch):

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.network.state_dict()
            },
            os.path.join(self.checkpoints_path, self.model_params_subdir,
                         str(epoch) + ".pth"),
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.network.state_dict()
            },
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"),
        )

        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir,
                         str(epoch) + ".pth"),
        )
        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"),
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--points_batch", type=int, default=16384 // 3, help="point batch size")
    parser.add_argument("--nepoch", type=int, default=21000, help="number of epochs to train for")
    parser.add_argument("--conf", type=str, default="confs/setup_base_all_PCA.conf")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use [default: GPU auto]")
    parser.add_argument("--is_continue", default=False, action="store_true", help="continue")
    parser.add_argument("--timestamp", default="latest", type=str)
    parser.add_argument("--checkpoint", default="latest", type=str)
    parser.add_argument("--eval", default=False, action="store_true")
    parser.add_argument("--shape", type=str, default="unknow", help="shape_name")
    parser.add_argument("--type", type=str, default="unknow", help="type")
    parser.add_argument("--noise_degree", type=str, default="unknow", help="no_noise")
    parser.add_argument("--size", type=int, default="80", help="grid_size")
    parser.add_argument("--outdir", type=str, default="Yo-Need-Set-this", help="grid_size")
    parser.add_argument("--dataset", type=str, default="thingi", help="dataset")
    parser.add_argument("--debug", type=str, default="N", help="dataset")

    args = parser.parse_args()
    args.expname = "%s" % (args.shape)

    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order="memory", limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = args.gpu

    setup_seed(20)
    start_time_training = time.time()

    trainrunner = ReconstructionRunner(
        conf=args.conf,
        points_batch=args.points_batch,
        nepochs=args.nepoch,
        expname=args.expname,
        gpu_index=gpu,
        is_continue=args.is_continue,
        timestamp=args.timestamp,
        checkpoint=args.checkpoint,
        eval=args.eval,
        shape=args.shape,
        type=args.type,
        noise_degree=args.noise_degree,
        size=args.size,
        home_dir=args.outdir,
        dataset=args.dataset,
        debug=args.debug,
    )

    trainrunner.run()

    consuming_time = time.time() - start_time_training
    with open("consuming.txt", "a+") as f:
        f.write("v2 compare ---shape: %s, :cosuming time:%f\n" % (args.shape, consuming_time))
