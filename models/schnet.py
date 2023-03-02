from typing import Optional, Callable, Dict

import torch
import torch.nn as nn

from schnetpack.nn.base import Dense
from schnetpack.nn.base import Aggregate
from schnetpack import Properties
from schnetpack.nn.cutoff import HardCutoff
from schnetpack.nn.acsf import GaussianSmearing
from schnetpack.nn.neighbors import AtomDistances
from schnetpack.nn.activations import shifted_softplus


def load_schnet_model(
    valence_informed: Optional[str] = None,
    valence_alpha: float = 1.0,
    use_no_bond: bool = False,
    vector_coeff: bool = False,
    is_alpha_learnable: bool = False,
    targets=[],
    return_intermediate: bool = False,
):
    hidden_size = 64

    schnet = SchNetBase(
        n_atom_basis=64,
        n_filters=hidden_size,
        n_gaussians=300,
        n_interactions=3,
        valence_informed=valence_informed,
        valence_alpha=valence_alpha,
        use_no_bond=use_no_bond,
        vector_coeff=vector_coeff,
        is_alpha_learable=is_alpha_learnable,
        return_intermediate=return_intermediate,
    )

    atomwise = []
    for prop in targets:
        if prop.enabled:
            atomwise.append(
                spk.atomistic.Atomwise(
                    n_in=hidden_size, n_out=prop.out_dim, property=prop.name
                )
            )
    schnet = spk.AtomisticModel(representation=schnet, output_modules=atomwise)
    return schnet


class CFConv(nn.Module):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        n_filters (int): number of filter dimensions.
        n_out (int): number of output dimensions.
        filter_network (nn.Module): filter block.
        cutoff_network (nn.Module, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(
        self,
        n_in,
        n_filters,
        n_out,
        filter_network,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
        axis=2,
        valence_alpha: float = 1.0,
        valence_informed: Optional[str] = None,
        use_no_bond=False,
        is_alpha_learnable: bool = False,
        vector_coeff: bool = False,
    ):
        super(CFConv, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None, bonds=None):
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        # pass expanded interactomic distances through filter block
        W = self.filter_network(f_ij)

        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)
        # reshape y for element-wise multiplication by W
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        # element-wise multiplication, aggregating and Dense layer
        sum1 = y * W
        y = self.agg(sum1, pairwise_mask)
        return y


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_spatial_basis (int): number of input features of filter-generating networks.
        n_filters (int): number of filters used in continuous-filter convolution.
        cutoff (float): cutoff radius.
        cutoff_network (nn.Module, optional): cutoff layer.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        n_atom_basis,
        n_spatial_basis,
        n_filters,
        cutoff,
        cutoff_network=HardCutoff,
        normalize_filter=False,
        valence_alpha: float = 1.0,
        valence_informed: Optional[str] = None,
        use_no_bond: bool = False,
        vector_coeff: bool = False,
        is_alpha_learable: bool = False,
    ):
        super(SchNetInteraction, self).__init__()
        # filter block used in interaction block
        self.n_bonds = 5 if use_no_bond else 4
        self.filter_network = nn.Sequential(
            Dense(n_spatial_basis, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters),
        )

        if valence_informed is not None and valence_informed == "smp":
            self.cfconv = []
            for _ in range(self.n_bonds + 1):
                filter_network = nn.Sequential(
                    Dense(n_spatial_basis, n_filters, activation=shifted_softplus),
                    Dense(n_filters, n_filters),
                )
                self.cfconv.append(
                    CFConv(
                        n_atom_basis,
                        n_filters,
                        n_atom_basis,
                        filter_network,
                        cutoff_network=cutoff_network(cutoff),
                        activation=shifted_softplus,
                        normalize_filter=normalize_filter,
                        valence_alpha=valence_alpha,
                        valence_informed=valence_informed,
                        use_no_bond=use_no_bond,
                        vector_coeff=vector_coeff,
                    )
                )
            self.cfconv = nn.ModuleList(self.cfconv)
        else:
            filter_network = nn.Sequential(
                Dense(n_spatial_basis, n_filters, activation=shifted_softplus),
                Dense(n_filters, n_filters),
            )
            self.cfconv = CFConv(
                n_atom_basis,
                n_filters,
                n_atom_basis,
                filter_network,
                cutoff_network=cutoff_network(cutoff),
                activation=shifted_softplus,
                normalize_filter=normalize_filter,
                valence_alpha=valence_alpha,
                valence_informed=valence_informed,
                use_no_bond=use_no_bond,
                vector_coeff=vector_coeff,
            )
        self.valence_informed = valence_informed

        if vector_coeff:
            self.bond_coef = nn.Parameter(
                torch.randn(self.n_bonds, 64), requires_grad=True
            )
        else:
            self.bond_coef = nn.Parameter(torch.randn(self.n_bonds), requires_grad=True)

        self.valence_alpha = torch.tensor(valence_alpha)
        if is_alpha_learable:
            self.valence_alpha = nn.Parameter(self.valence_alpha, requires_grad=True)

        if valence_informed and valence_informed == "suf":
            self.f2out = nn.ModuleList(
                [
                    Dense(
                        n_filters, n_atom_basis, bias=True, activation=shifted_softplus
                    )
                    for _ in range(self.n_bonds + 1)
                ]
            )
            self.dense = nn.ModuleList(
                [
                    Dense(n_filters, n_atom_basis, bias=True, activation=None)
                    for _ in range(self.n_bonds + 1)
                ]
            )
        else:
            self.f2out = Dense(
                n_filters, n_atom_basis, bias=True, activation=shifted_softplus
            )
            self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def specialised_message_production(
        self, x, r_ij, neighbors, neighbor_mask, f_ij, bonds
    ):
        v = self.cfconv[0](
            x, r_ij, neighbors, neighbor_mask, f_ij, bonds
        )  # generic message
        v_b = []
        va = self.valence_alpha.clamp(0, 1)
        for bt in range(1, self.n_bonds + 1):
            neighbors_n = (bonds[bt - 1] * neighbors).long()
            v_b.append(
                self.cfconv[bt](x, r_ij, neighbors_n, neighbor_mask, f_ij, bonds)
            )
        v_b = torch.stack(v_b).sum(0, keepdim=False)
        v = va * v + (1 - va) * v_b
        return v

    def specialised_weighted_message(
        self, x, r_ij, neighbors, neighbor_mask, f_ij, bonds
    ):
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij, bonds)
        v_b = []
        va = self.valence_alpha.clamp(0, 1)
        for bt in range(self.n_bonds):
            neighbors_n = (bonds[bt - 1] * neighbors).long()
            v_b.append(
                self.bond_coef[bt]
                * self.cfconv(x, r_ij, neighbors_n, neighbor_mask, f_ij, bonds)
            )
        v_b = torch.stack(v_b).sum(0, keepdim=False)
        v = va * v + (1 - va) * v_b
        return v

    def specialised_update_function_phase(
        self, x, r_ij, neighbors, neighbor_mask, f_ij, bonds
    ):
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij, bonds)
        v_b = []
        for bt in range(self.n_bonds):
            neighbors_n = (bonds[bt - 1] * neighbors).long()
            v_b.append(self.cfconv(x, r_ij, neighbors_n, neighbor_mask, f_ij, bonds))
        return v, v_b

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None, bonds=None):
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_atom_basis) shape.

        """
        # continuous-filter convolution interaction block followed by Dense layer
        if self.valence_informed is not None:
            if self.valence_informed == "smp":
                v = self.specialised_message_production(
                    x, r_ij, neighbors, neighbor_mask, f_ij, bonds
                )
            elif self.valence_informed == "swm":
                v = self.specialised_weighted_message(
                    x, r_ij, neighbors, neighbor_mask, f_ij, bonds
                )
            elif self.valence_informed == "suf":
                v, v_b = self.specialised_update_function_phase(
                    x, r_ij, neighbors, neighbor_mask, f_ij, bonds
                )
        else:
            v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij, bonds)

        if self.valence_informed is not None and self.valence_informed == "suf":
            va = self.valence_alpha.clamp(0, 1)
            v_g = self.f2out[0](v)
            v_g = self.dense[0](v_g)
            u_b = []
            for bt in range(self.n_bonds):
                v_bs = v_b[bt]
                v_bs = self.f2out[bt + 1](v_bs)
                v_bs = self.dense[bt + 1](v_bs)
                u_b.append(v_bs)
            u_b = torch.stack(u_b).sum(0, keepdim=False)
            v = va * v + (1 - va) * u_b
        else:
            v = self.f2out(v)
            v = self.dense(v)
        return v


class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems.

    Args:
        n_atom_basis (int, optional): number of features to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        n_filters (int, optional): number of filters used in continuous-filter convolution
        n_interactions (int, optional): number of interaction blocks.
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool, optional): if True, `forward` method also returns
            intermediate atomic representations after each interaction block is applied.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
        cutoff_network (nn.Module, optional): cutoff layer.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.
        distance_expansion (nn.Module, optional): layer for expanding interatomic
            distances in a basis.
        charged_systems (bool, optional):

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis=128,
        n_filters=128,
        n_interactions=3,
        cutoff=5.0,
        n_gaussians=25,
        normalize_filter=False,
        coupled_interactions=False,
        return_intermediate=False,
        max_z=100,
        cutoff_network=HardCutoff,
        trainable_gaussians=False,
        distance_expansion=None,
        charged_systems=False,
        valence_informed: Optional[str] = None,
        valence_alpha: float = 1.0,
        is_alpha_learable: bool = False,
        use_no_bond: bool = False,
        vector_coeff: bool = False,
        return_hidden_states: bool = False,
    ):
        super().__init__()

        self.n_atom_basis = n_atom_basis
        # make a lookup table to store embeddings for each element (up
        # to atomic number max_z) each of which is a vector of size
        # n_atom_basis
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # layer for computing interatomic distances
        self.distances = AtomDistances()

        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            self.distance_expansion = GaussianSmearing(
                0.0, cutoff, n_gaussians, trainable=trainable_gaussians
            )
        else:
            self.distance_expansion = distance_expansion

        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                        valence_alpha=valence_alpha,
                        valence_informed=valence_informed,
                        use_no_bond=use_no_bond,
                        vector_coeff=vector_coeff,
                        is_alpha_learable=is_alpha_learable,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                        valence_alpha=valence_alpha,
                        valence_informed=valence_informed,
                        use_no_bond=use_no_bond,
                        vector_coeff=vector_coeff,
                        is_alpha_learable=is_alpha_learable,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # set attributes
        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)
        self.valence_informed = valence_informed
        self.valence_alpha = valence_alpha
        self.state_updater = StateUpdater()

    def forward(self, inputs):
        """Compute atomic representations/embeddings.

        Args: inputs (dict of torch.Tensor): SchNetPack dictionary of
            input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.

        """
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]
        bonds = inputs["bonds"] if self.valence_informed is not None else None

        # get atom embeddings for the input atomic numbers
        x = self.embedding(atomic_numbers)

        if False and self.charged_systems and Properties.charge in inputs.keys():
            n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
            charge = inputs[Properties.charge] / n_atoms  # B
            charge = charge[:, None] * self.charge  # B x F
            x = x + charge

        # compute interatomic distance of every atom to its neighbors
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)
        # store intermediate representations
        if self.return_intermediate:
            xs = [x]
        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij, bonds=bonds)
            x = self.state_updater(x, v)
            if self.return_intermediate:
                xs.append(x)

        if self.return_intermediate:
            return x, xs
        return x


class StateUpdater(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, v):
        return x + v
