# internal imports
from typing import Optional, List

# External imports
import torch
import torch.nn as nn
from torch.autograd import Variable

# Custom modules
from .MessageFunction import MessageFunction
from .UpdateFunction import UpdateFunction
from .ReadoutFunction import ReadoutFunction


class SpecialisedUpdateFunctionOption1(nn.Module):
    def __init__(self, ms, hs):
        super().__init__()
        # Define Update
        self.u = nn.ModuleList(
            [UpdateFunction("mpnn", args={"in_m": ms, "out": hs}) for _ in range(0, 5)]
        )

    def forward(self, g, m, b, h):
        h_tb = []
        for i in range(0, b.shape[-1]):
            m_b = (b[..., i] == 1).float().unsqueeze(3) * m  # get messages of bond type
            m_b = torch.unsqueeze(g, 3).expand_as(m_b) * m_b  # for only neighbours
            m_b = m_b.sum(1)  # sum over bonds and neighbours
            h_b = self.u[i + 1](h, m_b)  # apply specialised update function
            h_tb.append(h_b)
        h_tb = torch.stack(h_tb).sum(0)  # sum over specialised relations
        return h_tb


class SpecialisedUpdateFunctionOption2(nn.Module):
    def __init__(self, ms, hs):
        super().__init__()
        self.u = UpdateFunction("mpnn", args={"in_m": ms * 5, "out": hs})
        self.states = []

    def forward(self, g, m, b, h):
        m_tb = []
        for i in range(0, b.shape[-1]):
            m_b = (b[..., i] == 1).float().unsqueeze(3) * m  # get messages of bond type
            m_b = torch.unsqueeze(g, 3).expand_as(m_b) * m_b  # for only neighbours
            m_b = m_b.sum(1)  # sum over bonds and neighbours
            m_tb.append(m_b)
            self.states.append(m_b.cpu().detach())
        m_tb = torch.cat(m_tb, dim=2)  # concatenate into single tensor
        h_tb = self.u(h, m_tb)  # apply single specialised message
        return h_tb


class SpecialisedUpdateFunctionOption3(nn.Module):
    def __init__(self, ms, hs):
        super().__init__()
        self.u = UpdateFunction("mpnn", args={"in_m": ms, "out": hs})

    def forward(self, g, m, b, h):
        h_tb = []
        for i in range(0, b.shape[-1]):
            m_b = (b[..., i] == 1).float().unsqueeze(3) * m  # get messages of bond type
            m_b = torch.unsqueeze(g, 3).expand_as(m_b) * m_b  # for only neighbours
            m_b = m_b.sum(1)  # sum over bonds and neighbours
            h_b = self.u(h, m_b)
            h_tb.append(h_b)
        h_tb = torch.stack(h_tb).sum(0)
        return h_tb


class MPNN(nn.Module):
    """
    MPNN as proposed by Gilmer et al..

    This class implements the whole Gilmer et al. model following the functions
    Message, Update and Readout.

    Parameters
    ----------
    in_n : int list
        Sizes for the node and edge features.
    hidden_state_size : int
        Size of the hidden states (the input will be padded with 0's to this
        size).
    message_size : int
        Message function output vector size.
    n_layers : int
        Number of iterations Message+Update (weight tying).
    l_target : int
        Size of the output.
    type : str (Optional)
        Classification | [Regression (default)]. If classification,
        LogSoftmax layer is applied to the output vector.
    """

    def __init__(
        self,
        in_n: List[int] = [5, 5],
        hidden_state_size: int = 73,
        message_size: int = 73,
        n_layers: int = 3,
        l_target: int = 1,
        type: str = "regression",
        valence_alpha: float = 0.0,
        return_hidden_states: bool = False,
        valence_informed: str = "default",
        suf_option: Optional[str] = None,
        trainable_alpha: bool = True,
        scalar_bond: bool = True,
    ):
        super(MPNN, self).__init__()
        self.valence_informed = valence_informed
        self.messaging_phase = {
            "swm": self.specialised_weighted_message,
            "suf": self.specialised_update_function,
            "smp": self.specialised_message_production,
            "default": self.default_messaging_phase,
        }.get(self.valence_informed)

        assert self.messaging_phase is not None
        self.hss = hidden_state_size
        self.mss = message_size
        self.suf_option = suf_option

        if valence_informed == "smp":
            self.m = nn.ModuleList(
                [
                    MessageFunction(
                        "mpnn",
                        args={
                            "edge_feat": in_n[1],
                            "in": self.hss,
                            "out": self.mss,
                        },
                    )
                    for _ in range(0, 6)  # 5 bonds plus generic message
                ]
            )
        else:
            # Define message
            self.m = nn.ModuleList(
                [
                    MessageFunction(
                        "mpnn",
                        args={
                            "edge_feat": in_n[1],
                            "in": self.hss,
                            "out": self.mss,
                        },
                    )
                ]
            )

        # Define Update
        self.u = nn.ModuleList(
            [UpdateFunction("mpnn", args={"in_m": self.mss, "out": self.hss})]
        )

        self.u_b = None
        if suf_option == "1":
            self.u_b = SpecialisedUpdateFunctionOption1(self.mss, self.hss)
        if suf_option == "2":
            self.u_b = SpecialisedUpdateFunctionOption2(self.mss, self.hss)
        if suf_option == "3":
            self.u_b = SpecialisedUpdateFunctionOption3(self.mss, self.hss)

        # check if an suf option has been selected and initialised
        if suf_option is not None:
            assert self.u_b is not None

        # Define Readout
        self.r = ReadoutFunction(
            "mpnn",
            args={
                "in": self.hss,
                "target": l_target,
                "return_hidden_states": return_hidden_states,
            },
        )

        self.type = type
        self.return_hidden_states = return_hidden_states
        self.args = {"out": self.hss}
        self.n_layers = n_layers
        self.valence_alpha = torch.nn.Parameter(
            torch.tensor(valence_alpha),
            requires_grad=trainable_alpha,
        )
        self.n_atoms = ReadoutFunction("mpnn", args={"in": self.hss, "target": 7})
        self.n_electrons = ReadoutFunction("mpnn", args={"in": self.hss, "target": 7})
        self.pdf = ReadoutFunction("mpnn", args={"in": self.hss, "target": 1})
        self.bond_coef = nn.Parameter(
            torch.randn(5, 1 if scalar_bond else self.hss), requires_grad=True
        )

    def specialised_weighted_message(self, g, h_in, e, b, h, t, m, m_f):
        bond_coef = b @ self.bond_coef
        m_tb = g.unsqueeze(3).expand_as(m) * (bond_coef * m)
        m_tb = m_tb.sum(1).squeeze(1)

        va = torch.sigmoid(self.valence_alpha)
        m = va * m_f + (1 - va) * m_tb
        h_t = self.u[0].forward(h[t], m)
        return h_t

    def specialised_update_function(self, g, h_in, e, b, h, t, m, m_f):
        # create generic update
        h_f = self.u[0].forward(h[t], m_f)

        h_tb = self.u_b(g, m, b, h[t])
        va = torch.sigmoid(self.valence_alpha)
        h_t = va * h_f + (1 - va) * h_tb
        return h_t

    def specialised_message_production(self, g, h_in, e, b, h, t, m, m_f):
        e_aux = e.view(-1, e.size(3))
        h_aux = h[t].view(-1, h[t].size(2))

        m_tb = []
        for b_t in range(0, b.shape[-1]):
            m_nr = self.m[b_t + 1](h[t], h_aux, e_aux)  # apply specialised message
            m_nr = m_nr.view(
                h[0].size(0), h[0].size(1), -1, m_nr.size(1)
            )  # reshape  [b, n, n, dim]
            m_nr = (b[..., b_t] == 1).float().unsqueeze(
                3
            ) * m_nr  # get only messages of bond type
            m_nr = g.unsqueeze(3).expand_as(m_nr) * m_nr  # for only neighbours
            m_tb.append(m_nr)
        m_tb = torch.stack(m_tb).sum(2)  # sum over neighbours
        m_tb = m_tb.sum(0)  # sum over bonds

        va = torch.sigmoid(self.valence_alpha)
        m = va * m_f + (1 - va) * m_tb
        h_t = self.u[0].forward(h[t], m)
        return h_t

    def default_messaging_phase(self, g, h_in, e, b, h, t, m, m_f):
        h_t = self.u[0].forward(h[t], m_f)
        return h_t  # no special change to message

    def forward(self, g, h_in, e, b):

        h = []

        # Padding to some larger dimension d
        h_t = torch.cat(
            [
                h_in,
                Variable(
                    torch.zeros(
                        h_in.size(0), h_in.size(1), self.args["out"] - h_in.size(2)
                    ).type_as(h_in.data)
                ),
            ],
            2,
        )

        h.append(h_t.clone())

        # Layer
        for t in range(0, self.n_layers):
            e_aux = e.view(-1, e.size(3))
            h_aux = h[t].view(-1, h[t].size(2))

            m = self.m[0].forward(h[t], h_aux, e_aux)
            m = m.view(h[0].size(0), h[0].size(1), -1, m.size(1))

            m_f = torch.unsqueeze(g, 3).expand_as(m) * m
            m_f = torch.squeeze(torch.sum(m_f, 1), dim=1)

            h_t = self.messaging_phase(g, h_in, e, b, h, t, m, m_f)

            # Delete virtual nodes
            h_t = (torch.sum(h_in, 2, keepdim=True).expand_as(h_t) > 0).type_as(
                h_t
            ) * h_t
            h.append(h_t)

        # Readout
        res, nn_res = self.r.forward(h)
        n_atoms = torch.zeros(res.size(0), 7)
        n_electrons = torch.zeros(res.size(0), 7)
        pdf = torch.zeros(res.size(0), 1)

        if hasattr(self, "n_atoms"):
            n_atoms, _ = self.n_atoms(h)
        if hasattr(self, "n_electrons"):
            n_electrons, _ = self.n_electrons(h)
        if hasattr(self, "pdf"):
            pdf, _ = self.pdf(h)

        if self.return_hidden_states:
            return res, n_atoms, n_electrons, pdf, nn_res

        return res, n_atoms, n_electrons, pdf
