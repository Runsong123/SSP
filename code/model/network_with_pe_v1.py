import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad
from model.embedding import *


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True,
                       only_inputs=True)[0][:, -3:]
    return points_grad


class ImplicitNet(nn.Module):
    def __init__(self, d_in, dims, skip_in=(), geometric_init=True, radius_init=1, beta=100, multires=6):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        # print(input_ch)
        self.current_multires = 0
        # mask = 3 * 3 * 2
        self.mask = torch.zeros([1, 39]).cuda()
        # self.mask[:, :3 + 6 * self.current_multires] = 1

        # exit()

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                elif multires > 0 and layer == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and layer in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            # if true preform preform geometric initialization
            # if geometric_init:

            #     if layer == self.num_layers - 2:

            #         torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
            #         torch.nn.init.constant_(lin.bias, -radius_init)
            #     else:
            #         torch.nn.init.constant_(lin.bias, 0.0)

            #         torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def set_mask(self, scale):
        self.current_multires = scale
        print("current_scale", self.current_multires)
        self.mask[:, :3 + 6 * self.current_multires] = 1

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        # x = input
        input = input * self.mask
        x = input
        # exit()
        # x = x * self.mask
        # mask = torch.zeros([])

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
