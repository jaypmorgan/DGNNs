import torch.nn as nn
import torch.nn.functional as F

class NNet(nn.Module):
    def __init__(
        self, n_in, n_out, hlayers=(128, 256, 128), return_transformations=False
    ):
        super(NNet, self).__init__()
        self.n_hlayers = len(hlayers)
        self.return_transformations = return_transformations
        self.fcs = nn.ModuleList(
            [
                nn.Linear(n_in, hlayers[i])
                if i == 0
                else nn.Linear(hlayers[i - 1], n_out)
                if i == self.n_hlayers
                else nn.Linear(hlayers[i - 1], hlayers[i])
                for i in range(self.n_hlayers + 1)
            ]
        )

    def forward(self, x):
        x_transformations = []
        x = x.contiguous().view(-1, self.num_flat_features(x))
        for i in range(self.n_hlayers):
            x = F.relu(self.fcs[i](x))
            x_transformations.append(x)
        x = self.fcs[-1](x)
        x_transformations.append(x)
        if self.return_transformations:
            return x, x_transformations
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
