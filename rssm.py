import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_


class Deter(nn.Module):
    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        self.hidden = int(hidden)
        if deter % self.blocks:
            raise ValueError(f"deter ({deter}) must be divisible by blocks ({self.blocks})")
        if self.hidden % self.blocks:
            raise ValueError(f"hidden ({self.hidden}) must be divisible by blocks ({self.blocks})")
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def _project_inputs(self, stoch, deter, action):
        """Project the recurrent state, stochastic state, and bounded action."""
        B = action.shape[0]
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)
        return x0, x1, x2

    def _gate_preactivations(self, deter, x0, x1, x2):
        """Compute the existing DreamerV3 block-GRU gate preactivations."""
        x = torch.cat([x0, x1, x2], -1)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))
        x = self._dyn_hid(x)
        x = self._dyn_gru(x)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)
        return tuple(self.group2flat(gate) for gate in gates)

    def _candidate(self, reset, cand, stoch_input, action_input):
        return torch.tanh(reset * cand)

    def forward(self, stoch, deter, action):
        """Deterministic state transition (block-GRU style)."""
        # (B, S, K), (B, D), (B, A)
        x0, x1, x2 = self._project_inputs(stoch, deter, action)
        reset, cand, update = self._gate_preactivations(deter, x0, x1, x2)
        reset = torch.sigmoid(reset)
        cand = self._candidate(reset, cand, x1, x2)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class SRUDeter(Deter):
    """DreamerV3 block-GRU enhanced with the SRU spatial transform.

    The paper's current recurrent input is represented here by the learned
    stochastic-state and action projections. The previous deterministic state
    remains part of the ordinary GRU gates but is deliberately excluded from
    the input-only spatial transformation.
    """

    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super().__init__(deter, stoch, act_dim, hidden, blocks, dynlayers, act)
        self._spatial_transform = BlockLinear(2 * self.hidden, deter, self.blocks)

    def _spatial_term(self, stoch_input, action_input):
        # Pair corresponding stochastic and action slices so each projection
        # block sees both parts of the current transition-input latent.
        grouped = torch.cat(
            [self.flat2group(stoch_input), self.flat2group(action_input)], dim=-1
        )
        return self._spatial_transform(self.group2flat(grouped))

    def _candidate(self, reset, cand, stoch_input, action_input):
        spatial = self._spatial_term(stoch_input, action_input)
        return torch.tanh(spatial * reset * cand)


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self._recurrent = str(getattr(config, "recurrent", "gru"))
        recurrent_cores = {"gru": Deter, "sru": SRUDeter}
        if self._recurrent not in recurrent_cores:
            choices = ", ".join(recurrent_cores)
            raise ValueError(f"rssm.recurrent must be one of {{{choices}}}, got {self._recurrent!r}")
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        self._deter_net = recurrent_cores[self._recurrent](
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
        )

        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        self.apply(weight_init_)

    def initial(self, batch_size):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations."""
        # (B, T, E), (B, T, A), ((B, S, K), (B, D)) (B, T)
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        """Single posterior step."""
        # (B, S, K), (B, D), (B, A), (B, E), (B,)
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

        # Deterministic transition then posterior logits conditioned on embed.
        # (B, D)
        deter = self._deter_net(stoch, deter, prev_action)
        # (B, D + E)
        x = torch.cat([deter, embed], dim=-1)
        # (B, S, K)
        logit = self._obs_net(x)

        # Sample discrete stochastic state via straight-through Gumbel-Softmax.
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        """Single prior step (no observation)."""

        # (B, D)
        deter = self._deter_net(stoch, deter, prev_action)
        # (B, S, K)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def prior(self, deter):
        """Compute prior distribution parameters and sample stoch."""

        # (B, S, K)
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions):
        """Roll out prior dynamics given a sequence of actions."""
        # (B, S, K), (B, D), (B, T, A)
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        # (B, T, S, K), (B, T, D)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        # (B, S, K), (B, D)
        # (B, S*K)
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        # (B, S*K + D)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        # Clipped gradients are not backpropagated using torch.clip.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss
