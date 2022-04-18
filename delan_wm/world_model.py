import torch
import torch.nn as nn
from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork


def add_layer(input_shape, hidden_shape, output_shape):
    return nn.Sequential(nn.Linear(input_shape, hidden_shape),
                         nn.Softplus(),
                         nn.Linear(hidden_shape, output_shape),
                         nn.Softplus()
                         )


class WorldModel(nn.Module):

    def __init__(self, args):
        super(WorldModel, self).__init__()
        self.delan = DeepLagrangianNetwork(args)

        self.q_encoder = add_layer(args.pos_shape + args.rew_shape, args.n_width, args.dof)
        self.qd_encoder = add_layer(args.vel_shape + args.rew_shape, args.n_width, args.dof)
        self.tau_encoder = add_layer(args.tau_shape + args.rew_shape, args.n_width, args.dof)
        # *3 for q,qd,qdd
        self.pred_q = add_layer(args.dof * 3, args.n_width, args.dof)
        self.pred_qd = add_layer(args.dof * 3, args.n_width, args.dof)

        self.pred_tau = add_layer(args.dof * 3, args.n_width, args.tau_shape - args.dof)  # have action in tau
        for name, param in self.named_parameters():
            if len(param.shape) >= 2:
                torch.nn.init.xavier_normal_(param, args.gain_hidden)
            elif 'bias' in name:
                param.data.fill_(0.)
        self.n_dof = args.dof
        self.activation = nn.Softplus()
        self.q_bn = nn.BatchNorm1d(args.pos_shape)
        self.qd_bn = nn.BatchNorm1d(args.vel_shape)
        # self.tau_bn=nn.BatchNorm1d(args.tau_shape)
        self.tau_bn = nn.BatchNorm1d(24)
        self.rew_bn = nn.BatchNorm1d(args.rew_shape)  ## not implemented, little tricky

    def forward(self, q, qd, contact_force, action_force, rew_state):
        # input -> q, qd, force
        q, qd, rew_state, contact_force = self.q_bn(q), self.qd_bn(qd), self.rew_bn(rew_state), self.tau_bn(
            contact_force)
        tau = torch.cat([contact_force, action_force], dim=1)
        q, qd, tau = self.q_encoder(torch.cat([q, rew_state], dim=-1)), \
                     self.qd_encoder(torch.cat([qd, rew_state], dim=-1)), \
                     self.tau_encoder(torch.cat([tau, rew_state], dim=-1))

        # delan model: for_dyn & energy_dot
        out = self.delan._dyn_model(q, qd, torch.zeros_like(q))  ### tau is 0 in delan-forward dynamic
        H, c, g = out[1], out[2], out[3]
        invH = torch.inverse(H)
        # below: actually F=ma
        qdd_pred = torch.matmul(invH, (tau - c - g).view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        dEdt = out[6] + out[7]

        forward_var=torch.cat([q,qd,qdd_pred],dim=-1)
        q_next = self.pred_q(forward_var) * self.q_bn.running_var + self.q_bn.running_mean
        qd_next = self.pred_qd(forward_var) * self.qd_bn.running_var + self.qd_bn.running_mean
        tau_next = self.pred_tau(forward_var) * self.tau_bn.running_var + self.tau_bn.running_mean
        return q_next, qd_next, dEdt, tau_next
