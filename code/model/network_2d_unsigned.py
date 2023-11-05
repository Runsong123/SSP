import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad
import torch.nn.functional as F


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True, only_inputs=True
    )[0][:, -2:]
    return points_grad


class ImplicitNet(nn.Module):
    def __init__(self, d_in, dims, skip_in=(), geometric_init=True, radius_init=1, beta=100):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization

            setattr(self, "lin" + str(layer), lin)

        # if beta > 0:
        #     self.activation = nn.Softplus(beta=beta)

        # # vanilla relu
        # else:
        self.activation = nn.ReLU()

    def forward(self, input):

        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            
            if layer < self.num_layers - 1:
                x = self.activation(x)
        return x
