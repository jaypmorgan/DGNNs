import os
import os.path as osp
from math import pi as PI
from math import sqrt
from typing import Callable, Optional, Tuple, Union, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding, Linear
from torch_sparse import SparseTensor

from torch_geometric.data import Dataset, download_url
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter
import torch_scatter as sct

qm9_target_dict = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    5: 'r2',
    6: 'zpve',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'G',
    11: 'Cv',
}


class Envelope(torch.nn.Module):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: Tensor) -> Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1. / x + a * x_pow_p0 + b * x_pow_p1 +
                c * x_pow_p2) * (x < 1.0).to(x.dtype)


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial: int, cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        dist = (dist.unsqueeze(-1) / self.cutoff)
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical: int, num_radial: int,
                 cutoff: float = 5.0, envelope_exponent: int = 5):
        super().__init__()
        import sympy as sym

        from torch_geometric.nn.models.dimenet_utils import (
            bessel_basis,
            real_sph_harm,
        )

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist: Tensor, angle: Tensor, idx_kj: Tensor) -> Tensor:
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_bilinear: int,
                 num_spherical: int, num_radial: int, num_before_skip: int,
                 num_after_skip: int, act: Callable):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = Linear(num_spherical * num_radial, num_bilinear,
                              bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(
            torch.Tensor(hidden_channels, num_bilinear, hidden_channels))

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor) -> Tensor:
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class InteractionPPBlockSMP(torch.nn.Module):
    def __init__(self, hidden_channels: int, int_emb_size: int,
                 basis_emb_size: int, num_spherical: int, num_radial: int,
                 num_before_skip: int, num_after_skip: int, act: Callable):
        super().__init__()
        self.act = act
        num_BT = 5
        self.bt_list = list(range(-1, num_BT))

        # Transformation of Bessel and spherical basis representations: ## \bar{M}
        self.lin_rbf1s = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lin_rbf1s.append(Linear(num_radial, basis_emb_size, bias=False))
        self.lin_rbf2s = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lin_rbf2s.append(Linear(basis_emb_size, hidden_channels, bias=False))

        self.lin_sbf1s = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lin_sbf1s.append(Linear(num_spherical * num_radial, basis_emb_size, bias=False))
        self.lin_sbf2s = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lin_sbf2s.append(Linear(basis_emb_size, int_emb_size, bias=False))

        # Hidden transformation of input message:
        self.lin_kjs = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lin_kjs.append(Linear(hidden_channels, hidden_channels)) ## \bar{M}
        self.lin_ji = Linear(hidden_channels, hidden_channels) ## \bar{U}

        # Embedding projections for interaction triplets:
        self.lin_downs = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lin_downs.append(Linear(hidden_channels, int_emb_size, bias=False)) ## \bar{M}
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False) ## \bar{U}

        # Residual layers before and after skip connection: ## \bar{U}
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin_rbf1 in self.lin_rbf1s:
            glorot_orthogonal(lin_rbf1.weight, scale=2.0)
        for lin_rbf2 in self.lin_rbf2s:
            glorot_orthogonal(lin_rbf2.weight, scale=2.0)
        for lin_sbf1 in self.lin_sbf1s:
            glorot_orthogonal(lin_sbf1.weight, scale=2.0)
        for lin_sbf2 in self.lin_sbf2s:
            glorot_orthogonal(lin_sbf2.weight, scale=2.0)

        for lin_kj in self.lin_kjs:
            glorot_orthogonal(lin_kj.weight, scale=2.0)
            lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        for lin_down in self.lin_downs:
            glorot_orthogonal(lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor, bt: Tensor, lambda_d, alpha) -> Tensor:
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x)) ## \bar{U}
        x_kj_spe = []
        for lin_kj, lin_rbf1, lin_rbf2, lin_down, lin_sbf1, lin_sbf2 in zip(self.lin_kjs, self.lin_rbf1s, self.lin_rbf2s, self.lin_downs, self.lin_sbf1s, self.lin_sbf2s):
            tmp = self.act(lin_kj(x)) ## \bar{M}

            # Transformation via Bessel basis:  ## \bar{M}
            rbf_tmp = lin_rbf1(rbf)
            rbf_tmp = lin_rbf2(rbf_tmp)
            tmp = tmp * rbf_tmp

            # Down project embedding and generating triple-interactions: ## \bar{M}
            tmp = self.act(lin_down(tmp))

            # Transform via 2D spherical basis: ## \bar{M}
            sbf_tmp = lin_sbf1(sbf)
            sbf_tmp = lin_sbf2(sbf_tmp)
            x_kj_spe.append(tmp[idx_kj] * sbf_tmp) ## embedding of edge kj for each triplet

        # Aggregate interactions and up-project embeddings:
        alpha = alpha if bt is not None else 1.0
        x_kj_tot = alpha * scatter(x_kj_spe[-1], idx_ji, dim=0, dim_size=x.size(0), reduce='sum') ## produces general \bar{m}_{vw}

        if bt is not None:
            bt_triplet = bt[idx_kj] ## bt for edge kj of each triplet
            for b in range(len(self.bt_list)):
                # if bt_list[b] == -1:  # treat no-bond as a special type of bond (which is why this is commented out)
                #     continue
                ind_triplet = torch.nonzero(bt_triplet == self.bt_list[b]).squeeze() ## index of triplets to keep for this sum (because edge kj is of BT b)
                out = scatter(x_kj_spe[b][ind_triplet], idx_ji[ind_triplet], dim_size=x.size(0), dim=0, reduce='sum')
                ind_not_covered = set(range(x.size(0))) - set(idx_ji[ind_triplet])
                ind_not_covered = list(ind_not_covered).sort()
                out[ind_not_covered] = 0
                x_kj_tot = x_kj_tot + (1-alpha) * out

        x_kj = self.act(self.lin_up(x_kj_tot)) ## \bar{U}

        ## \bar{U}
        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h ## \bar{h}_{vw}

class InteractionPPBlockSUF(torch.nn.Module):
    def __init__(self, hidden_channels: int, int_emb_size: int,
                 basis_emb_size: int, num_spherical: int, num_radial: int,
                 num_before_skip: int, num_after_skip: int, act: Callable):
        super().__init__()
        self.act = act
        num_BT = 5
        self.bt_list = list(range(-1, num_BT))

        # Transformation of Bessel and spherical basis representations: ## \bar{M}
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels) ## \bar{M}
        self.lin_jis = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lin_jis.append(Linear(hidden_channels, hidden_channels)) ## \bar{U}

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False) ## \bar{M}
        self.lin_ups = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lin_ups.append(Linear(int_emb_size, hidden_channels, bias=False)) ## \bar{U}

        # Residual layers before and after skip connection: ## \bar{U}
        self.layers_before_skips = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.layers_before_skips.append(torch.nn.ModuleList([
                ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
            ]))
        self.lins = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.layers_after_skips = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.layers_after_skips.append(torch.nn.ModuleList([
                ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
            ]))

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        for lin_ji in self.lin_jis:
            glorot_orthogonal(lin_ji.weight, scale=2.0)
            lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        for lin_up in self.lin_ups:
            glorot_orthogonal(lin_up.weight, scale=2.0)

        for layers_before_skip in self.layers_before_skips:
            for res_layer in layers_before_skip:
                res_layer.reset_parameters()
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        for layers_after_skip in self.layers_after_skips:
            for res_layer in layers_after_skip:
                res_layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor, bt: Tensor, alpha, lambda_d) -> Tensor:
        # Initial transformation:
        x_ji_spe = []
        for lin_ji in self.lin_jis:
            x_ji_spe.append(self.act(lin_ji(x))) ## \bar{U}
        x_kj = self.act(self.lin_kj(x)) ## \bar{M}

        # Transformation via Bessel basis:  ## \bar{M}
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions: ## \bar{M}
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis: ## \bar{M}
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf ## embedding of edge kj for each triplet

        # Aggregate interactions and up-project embeddings:
        x_kj_gen = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum') ## produces general \bar{m}_{vw}
        
        x_kj_gen = self.act(self.lin_ups[-1](x_kj_gen)) ## \bar{U}

        ## \bar{U}
        h_gen = x_ji_spe[-1] + x_kj_gen
        for layer in self.layers_before_skips[-1]:
            h_gen = layer(h_gen)
        h_gen = self.act(self.lins[-1](h_gen)) + x
        for layer in self.layers_after_skips[-1]:
            h_gen = layer(h_gen)

        alpha = alpha if bt is not None else 1.0
        h_tot = alpha * h_gen
        
        if bt is not None:
            bt_triplet = bt[idx_kj] ## bt for edge kj of each triplet

            for b in range(len(self.bt_list)):
                # if b == -1:  # treat no-bond as a special type of bond (which is why this is commented out)
                #     continue
                ind_triplet = torch.nonzero(bt_triplet == self.bt_list[b]).squeeze() ## index of triplets to keep for this sum (because edge kj is of BT b)
                out = scatter(x_kj[ind_triplet], idx_ji[ind_triplet], dim_size=x.size(0), dim=0, reduce='sum')
                ind_not_covered = set(range(x.size(0))) - set(idx_ji[ind_triplet])
                ind_not_covered = list(ind_not_covered).sort()
                out[ind_not_covered] = 0

                x_kj_spe = self.act(self.lin_ups[b](out)) ## \bar{U}

                ## \bar{U}
                h_spe = x_ji_spe[b] + x_kj_spe
                for layer in self.layers_before_skips[b]:
                    h_spe = layer(h_spe)
                h_spe = self.act(self.lins[b](h_spe)) + x
                for layer in self.layers_after_skips[b]:
                    h_spe = layer(h_spe)
                
                h_tot = h_tot + (1-alpha) * h_spe
                
        return h_tot ## \bar{h}_{vw}


class OutputPPBlockSUF(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int,
                 out_emb_channels: int, out_channels: int, num_layers: int,
                 act: Callable, num_aux: int = 0, aux_output_channels = None):
        super().__init__()
        self.act = act
        self.num_aux = num_aux

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False) ## M

        # The up-projection layer: ## readout
        self.lin_ups = torch.nn.ModuleList()
        for _ in range(num_aux+1):
            self.lin_ups.append(Linear(hidden_channels, out_emb_channels, bias=False))
        self.linss = torch.nn.ModuleList()
        for a in range(num_aux + 1):
            self.linss.append(torch.nn.ModuleList())
            for _ in range(num_layers):
                self.linss[a].append(Linear(out_emb_channels, out_emb_channels))
        self.lins = torch.nn.ModuleList()
        if aux_output_channels is None:
            aux_output_channels = [out_channels]*num_aux  # default all the same size
        aux_output_channels = [out_channels] + aux_output_channels
        for a in range(num_aux + 1):
            self.lins.append(Linear(out_emb_channels, aux_output_channels[a], bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin_up in self.lin_ups:
            glorot_orthogonal(lin_up.weight, scale=2.0)
        for lins in self.linss:
            for lin in lins:
                glorot_orthogonal(lin.weight, scale=2.0)
                lin.bias.data.fill_(0)
        for lin in self.lins:
            lin.weight.data.fill_(0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None, bt: Tensor = None,
                alpha = 1.0, lambda_d = None) -> Tensor:
        x = self.lin_rbf(rbf) * x ## M : message m_ji for each edge ji

        alpha = alpha if bt is not None else 1.0
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum') ## produces general m_v

        ## readout
        x_out = []
        for a in range(self.num_aux+1):
            x_out.append(self.lin_ups[a](x))
            for lin in self.linss[a]:
                x_out[a] = self.act(lin(x_out[a]))
            x_out[a] = self.lins[a](x_out[a])
        return x_out
    

class InteractionPPBlockSWM(torch.nn.Module):
    def __init__(self, hidden_channels: int, int_emb_size: int,
                 basis_emb_size: int, num_spherical: int, num_radial: int,
                 num_before_skip: int, num_after_skip: int, act: Callable):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations: ## \bar{M}
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels) ## \bar{M}
        self.lin_ji = Linear(hidden_channels, hidden_channels) ## \bar{U}

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False) ## \bar{M}
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False) ## \bar{U}

        # Residual layers before and after skip connection: ## \bar{U}
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor, bt: Tensor, lambda_d, alpha) -> Tensor:
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x)) ## \bar{U}
        x_kj = self.act(self.lin_kj(x)) ## \bar{M}

        # Transformation via Bessel basis:  ## \bar{M}
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions: ## \bar{M}
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis: ## \bar{M}
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf ## embedding of edge kj for each triplet

        # Aggregate interactions and up-project embeddings:
        alpha = alpha if bt is not None else 1.0
        x_kj_spe = alpha * scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum') ## produces general \bar{m}_{vw}

        if bt is not None:
            bt_triplet = bt[idx_kj] ## bt for edge kj of each triplet
            bt_list = torch.unique(bt_triplet) ## list of possible bound types

            for b in bt_list:
                # if b == -1:  # treat no-bond as a special type of bond (which is why this is commented out)
                #     continue
                ind_triplet = torch.nonzero(bt_triplet == b).squeeze() ## index of triplets to keep for this sum (because edge kj is of BT b)
                out = scatter(x_kj[ind_triplet], idx_ji[ind_triplet], dim_size=x.size(0), dim=0, reduce='sum')
                ind_not_covered = set(range(x.size(0))) - set(idx_ji[ind_triplet])
                ind_not_covered = list(ind_not_covered).sort()
                out[ind_not_covered] = 0
                x_kj_spe = x_kj_spe + (1-alpha) * lambda_d[b] * out

        x_kj = self.act(self.lin_up(x_kj_spe)) ## \bar{U}

        ## \bar{U}
        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h ## \bar{h}_{vw}


class InteractionPPBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, int_emb_size: int,
                 basis_emb_size: int, num_spherical: int, num_radial: int,
                 num_before_skip: int, num_after_skip: int, act: Callable):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations: ## \bar{M}
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels) ## \bar{M}
        self.lin_ji = Linear(hidden_channels, hidden_channels) ## \bar{U}

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False) ## \bar{M}
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False) ## \bar{U}

        # Residual layers before and after skip connection: ## \bar{U}
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor, bt: Tensor, lambda_d, alpha) -> Tensor:
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x)) ## \bar{U}
        x_kj = self.act(self.lin_kj(x)) ## \bar{M}

        # Transformation via Bessel basis:  ## \bar{M}
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions: ## \bar{M}
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis: ## \bar{M}
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf ## embedding of edge kj for each triplet

        # Aggregate interactions and up-project embeddings:
        alpha = alpha if bt is not None else 1.0
        x_kj_spe = alpha * scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum') ## produces general \bar{m}_{vw}

        x_kj = self.act(self.lin_up(x_kj_spe)) ## \bar{U}

        ## \bar{U}
        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h ## \bar{h}_{vw}


class OutputBlock(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int,
                 out_channels: int, num_layers: int, act: Callable):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class OutputPPBlockSWM(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int,
                 out_emb_channels: int, out_channels: int, num_layers: int,
                 act: Callable, num_aux: int = 0, aux_output_channels = None):
        super().__init__()
        self.act = act
        self.num_aux = num_aux

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False) ## M

        # The up-projection layer: ## readout
        self.lin_ups = torch.nn.ModuleList()
        for _ in range(num_aux+1):
            self.lin_ups.append(Linear(hidden_channels, out_emb_channels, bias=False))
        self.linss = torch.nn.ModuleList()
        for a in range(num_aux + 1):
            self.linss.append(torch.nn.ModuleList())
            for _ in range(num_layers):
                self.linss[a].append(Linear(out_emb_channels, out_emb_channels))
        self.lins = torch.nn.ModuleList()
        if aux_output_channels is None:
            aux_output_channels = [out_channels]*num_aux  # default all the same size
        aux_output_channels = [out_channels] + aux_output_channels
        for a in range(num_aux + 1):
            self.lins.append(Linear(out_emb_channels, aux_output_channels[a], bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin_up in self.lin_ups:
            glorot_orthogonal(lin_up.weight, scale=2.0)
        for lins in self.linss:
            for lin in lins:
                glorot_orthogonal(lin.weight, scale=2.0)
                lin.bias.data.fill_(0)
        for lin in self.lins:
            lin.weight.data.fill_(0)
        
    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None, bt: Tensor = None,
                lambda_d = None, alpha = 1.0) -> Tensor:
        x = self.lin_rbf(rbf) * x ## message m_ji for each edge ji

        alpha = alpha if bt is not None else 1.0
        x_spe = alpha * scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum') ## produces general m_v

        if bt is not None:
            lambda_d = lambda_d if lambda_d is not None else [1.0, 1.0, 1.0, 1.0]
            bt_list = torch.unique(bt) ## list of possible bound types

            for b in bt_list:
                # if b == -1:  # treat no-bond as a special type of bond (which is why this is commented out)
                #     continue
                ind_edge = torch.nonzero(bt == b).squeeze() ## index of edges to keep for this sum (because edge ji is of BT b)
                out = scatter(x[ind_edge], i[ind_edge], dim=0, dim_size=num_nodes, reduce='sum')
                ind_not_covered = set(range(num_nodes)) - set(i[ind_edge])
                ind_not_covered = list(ind_not_covered).sort()
                out[ind_not_covered] = 0
                x_spe = x_spe + (1 - alpha) * lambda_d[b] * out

        ## readout
        x_out = []
        for a in range(self.num_aux+1):
            x_out.append(self.lin_ups[a](x_spe))
            for lin in self.linss[a]:
                x_out[a] = self.act(lin(x_out[a]))
            x_out[a] = self.lins[a](x_out[a])
        return x_out


class OutputPPBlockSMP(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int,
                 out_emb_channels: int, out_channels: int, num_layers: int,
                 act: Callable, num_aux: int = 0, aux_output_channels = None):
        super().__init__()
        self.act = act
        num_BT = 5
        self.bt_list = list(range(-1, num_BT))
        self.num_aux = num_aux

        self.lin_rbfs = torch.nn.ModuleList()
        for _ in range(num_BT+1):
            self.lin_rbfs.append(Linear(num_radial, hidden_channels, bias=False)) ## M and M_r

        # The up-projection layer: ## readout
        self.num_aux = num_aux
        self.lin_ups = torch.nn.ModuleList()
        for _ in range(num_aux+1):
            self.lin_ups.append(Linear(hidden_channels, out_emb_channels, bias=False))
        self.linss = torch.nn.ModuleList()
        for a in range(num_aux + 1):
            self.linss.append(torch.nn.ModuleList())
            for _ in range(num_layers):
                self.linss[a].append(Linear(out_emb_channels, out_emb_channels))
        self.lins = torch.nn.ModuleList()
        if aux_output_channels is None:
            aux_output_channels = [out_channels]*num_aux  # default all the same size
        aux_output_channels = [out_channels] + aux_output_channels
        for a in range(num_aux + 1):
            self.lins.append(Linear(out_emb_channels, aux_output_channels[a], bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        for lin_rbf in self.lin_rbfs:
            glorot_orthogonal(lin_rbf.weight, scale=2.0)
        for lin_up in self.lin_ups:
            glorot_orthogonal(lin_up.weight, scale=2.0)
        for lins in self.linss:
            for lin in lins:
                glorot_orthogonal(lin.weight, scale=2.0)
                lin.bias.data.fill_(0)
        for lin in self.lins:
            lin.weight.data.fill_(0)
            
    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None, bt: Tensor = None,
                alpha = 1.0, lambda_d = None) -> Tensor:
        x_spe=[]
        for lin_rbf in self.lin_rbfs:
            x_spe.append(lin_rbf(rbf) * x) ## M : message m_ji for each edge ji

        alpha = alpha if bt is not None else 1.0
        x_tot = alpha * scatter(x_spe[-1], i, dim=0, dim_size=num_nodes, reduce='sum') ## produces general m_v

        if bt is not None:
            for b in range(len(self.bt_list)):
                # if bt_list[b] == -1:  # treat no-bond as a special type of bond (which is why this is commented out)
                #     continue
                ind_edge = torch.nonzero(bt == self.bt_list[b]).squeeze() ## index of edges to keep for this sum (because edge ji is of BT b)
                out = scatter(x_spe[b][ind_edge], i[ind_edge], dim=0, dim_size=num_nodes, reduce='sum')
                ind_not_covered = set(range(num_nodes)) - set(i[ind_edge])
                ind_not_covered = list(ind_not_covered).sort()
                out[ind_not_covered] = 0
                x_tot = x_tot + (1 - alpha) * out

        ## readout
        x_out = []
        for a in range(self.num_aux+1):
            x_out.append(self.lin_ups[a](x_tot))
            for lin in self.linss[a]:
                x_out[a] = self.act(lin(x_out[a]))
            x_out[a] = self.lins[a](x_out[a])
        return x_out


class OutputPPBlock(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int,
                 out_emb_channels: int, out_channels: int, num_layers: int,
                 act: Callable, num_aux: int = 0, aux_output_channels = None):
        super().__init__()
        self.act = act
        self.num_aux = num_aux

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False) ## M

        # The up-projection layer: ## readout
        self.lin_ups = torch.nn.ModuleList()
        for _ in range(num_aux+1):
            self.lin_ups.append(Linear(hidden_channels, out_emb_channels, bias=False))
        self.linss = torch.nn.ModuleList()
        for a in range(num_aux + 1):
            self.linss.append(torch.nn.ModuleList())
            for _ in range(num_layers):
                self.linss[a].append(Linear(out_emb_channels, out_emb_channels))
        self.lins = torch.nn.ModuleList()
        if aux_output_channels is None:
            aux_output_channels = [out_channels]*num_aux  # default all the same size
        aux_output_channels = [out_channels] + aux_output_channels
        for a in range(num_aux + 1):
            self.lins.append(Linear(out_emb_channels, aux_output_channels[a], bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin_up in self.lin_ups:
            glorot_orthogonal(lin_up.weight, scale=2.0)
        for lins in self.linss:
            for lin in lins:
                glorot_orthogonal(lin.weight, scale=2.0)
                lin.bias.data.fill_(0)
        for lin in self.lins:
            lin.weight.data.fill_(0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None, bt: Tensor = None,
                lambda_d = None, alpha = 1.0) -> Tensor:
        x = self.lin_rbf(rbf) * x ## message m_ji for each edge ji

        alpha = alpha if bt is not None else 1.0
        x_spe = alpha * scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum') ## produces general m_v

        ## readout
        x_out = []
        for a in range(self.num_aux+1):
            x_out.append(self.lin_ups[a](x))
            for lin in self.linss[a]:
                x_out[a] = self.act(lin(x_out[a]))
            x_out[a] = self.lins[a](x_out[a])
        return x_out


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class DimeNet(BaseModel):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.
    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    .. note::

        For an example of using a pretrained DimeNet variant, see
        `examples/qm9_pretrained_dimenet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_dimenet.py>`_.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
    """

    url = ('https://github.com/klicperajo/dimenet/raw/master/pretrained/'
           'dimenet')

    def __init__(
        self,
        hidden_channels: int = 128,
        num_blocks: int = 6,
        num_bilinear: int = 8,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        hparams = None,
        criterion = nn.MSELoss(),
    ):
        super().__init__(hparams=hparams)

        print(f"Using {cutoff} cutoff value")
        self.valence_alpha = hparams.valence_alpha
        self.valence_informed = hparams.valence_informed
        self.use_no_bond = True
        self.vector_coeff = hparams.vector_coeff
        self.is_alpha_learnable = hparams.is_alpha_learnable
        self.hidden_channels = hidden_channels

        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.params = hparams
        self.targets = hparams.targets
        self.criterion = criterion
        self.num_aux = len(self.targets)-1

        if num_spherical < 2:
            raise ValueError("num_spherical should be greater than 1")

        act = activation_resolver(act)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(num_radial, hidden_channels, len(self.targets),
                        num_output_layers, act) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(hidden_channels, num_bilinear, num_spherical,
                             num_radial, num_before_skip, num_after_skip, act)
            for _ in range(num_blocks)
        ])

        self.rg = True if self.params.valence_informed else False
        self.n_bonds = 5 if self.use_no_bond else 4
        if self.vector_coeff:
            self.lambda_d_out = nn.Parameter(torch.randn(self.n_bonds, self.hidden_channels), requires_grad=self.rg)
            self.lambda_d_int = nn.Parameter(torch.randn(self.n_bonds, self.hidden_channels//2), requires_grad=self.rg)
        else:
            self.lambda_d_out = nn.Parameter(torch.randn(self.n_bonds), requires_grad=self.rg)
            self.lambda_d_int = self.lambda_d_out
        self.alpha = torch.tensor(self.valence_alpha)
        if self.is_alpha_learnable:
            self.alpha = nn.Parameter(self.alpha, requires_grad=True)

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        if self.vector_coeff:
            self.lambda_d_out = nn.Parameter(torch.randn(self.n_bonds, self.hidden_channels), requires_grad=self.rg)
            self.lambda_d_int = nn.Parameter(torch.randn(self.n_bonds, self.hidden_channels//2), requires_grad=self.rg)
        else:
            self.lambda_d_out = nn.Parameter(torch.randn(self.n_bonds), requires_grad=self.rg)
            self.lambda_d_int = self.lambda_d_out
        if self.is_alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(self.valence_alpha), requires_grad=True)

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: OptTensor = None,
        bonds: Tensor = None,
    ) -> Tensor:
        """"""
        if batch is None and len(pos.shape) > 2:
            # the parameter is none so we need to make it itself. In
            # the DFTPrediction framework, even a batch size of 1 has
            # a batched dimension.
            if z.shape[0] == 1 and pos.shape[0] == 1:
                # we have a single batch than must be reduced
                z, pos = z.squeeze(0), pos.squeeze(0)
            else:
                # we have a batch of inputs which we need to
                # concatenate into a larger graph and provide a batch
                # tensor.
                batch_size, n_nodes, feature_size = pos.shape
                z, pos = z.reshape(-1), pos.reshape(-1, 3)
                # now create the batch tensor that enables the correct
                # indexing on this concatenated graph.
                batch = (torch
                         .arange(0, batch_size, device=pos.device)
                         .repeat_interleave(n_nodes))

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        if bonds is not None:
            # we need to create a bond-type vector the same size as
            # edge_index
            bt = torch.zeros_like(edge_index[1]).fill_(-1)  # -1=no-bond
            idx = 0
            for u, v in zip(edge_index[1], edge_index[0]):
                bt[idx] = bonds[0, u, v]
                idx += 1
        else:
            bt = None

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](
            x, rbf, i, num_nodes=pos.size(0),
            bt=bt, lambda_d=self.lambda_d_out, alpha=self.alpha)

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji,
                                  bt=bt, lambda_d=self.lambda_d_int, alpha=self.alpha)
            outs = output_block(x, rbf, i, num_nodes=pos.size(0),
                                 bt=bt, lambda_d=self.lambda_d_out, alpha=self.alpha)
            for a in range(len(outs)):
                P[a] = P[a] + outs[a]

        out_dict = {}
        for idx, target in enumerate(self.targets):
            out_dict[target.name] = P[idx].sum(dim=0).unsqueeze(0)
        return out_dict


class DimeNetPlusPlus(DimeNet):
    r"""The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    :class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model with
    8x faster and 10% more accurate than :class:`DimeNet`.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation funtion.
            (default: :obj:`"swish"`)
    """

    url = ('https://raw.githubusercontent.com/gasteigerjo/dimenet/'
           'master/pretrained/dimenet_pp')

    def __init__(
        self,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 4,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        *args,
        **kwargs,
    ):
        act = activation_resolver(act)

        super().__init__(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            num_bilinear=1,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            *args,
            **kwargs
        )

        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initalization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:

        OutBlock = {
            "swm": OutputPPBlockSWM,
            "smp": OutputPPBlockSMP,
            "suf": OutputPPBlockSUF,
        }.get(self.valence_informed, OutputPPBlock)

        auxiliary_output_sizes = None
        if self.num_aux:
            # build the size of auxiliary output channels
            auxiliary_output_sizes = []
            for target in self.targets:
                if target != "energy":
                    auxiliary_output_sizes.append(target.out_dim)

        self.output_blocks = torch.nn.ModuleList([
            OutBlock(num_radial, hidden_channels, out_emb_channels, 1,
                     num_output_layers, act, num_aux=self.num_aux,
                     aux_output_channels=auxiliary_output_sizes)
            for _ in range(num_blocks + 1)
        ])

        IntBlock = {
            "swm": InteractionPPBlockSWM,
            "smp": InteractionPPBlockSMP,
            "suf": InteractionPPBlockSUF,
        }.get(self.valence_informed, InteractionPPBlock)

        print(f"Using {IntBlock} due to {self.valence_informed} being selected")
        print(f"Using {OutBlock} due to {self.valence_informed} being selected")

        self.interaction_blocks = torch.nn.ModuleList([
            IntBlock(hidden_channels, int_emb_size, basis_emb_size,
                               num_spherical, num_radial, num_before_skip,
                               num_after_skip, act) for _ in range(num_blocks)
        ])

        self.reset_parameters()
