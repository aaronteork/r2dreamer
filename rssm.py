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
    """DreamerV3 block-GRU with proprioception-conditioned SRU spatial modulation.

    Predicts the forward proprioceptive transition from (deter, stoch, action)
    using Dreamer's standard MLP architecture (Linear -> RMSNorm -> act -> Linear -> RMSNorm -> act -> Linear).
    The predicted proprioception is projected to modulate the candidate state via
    BlockLinear before tanh in both observation and imagination rollouts.
    """

    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU", proprio_dim=26, spatial_modulation=True):
        super().__init__(deter, stoch, act_dim, hidden, blocks, dynlayers, act)
        self.deter = int(deter)
        self.proprio_dim = int(proprio_dim)
        self.spatial_modulation = bool(spatial_modulation)
        act_fn = getattr(torch.nn, act)

        # Forward proprioception predictor matching standard Dreamer MLP:
        # Linear -> RMSNorm -> act -> Linear -> RMSNorm -> act -> Linear
        self._proprio_pred = nn.Sequential(
            nn.Linear(3 * self.hidden, self.hidden, bias=True),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
            act_fn(),
            nn.Linear(self.hidden, self.hidden, bias=True),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
            act_fn(),
            nn.Linear(self.hidden, self.proprio_dim, bias=True),
        )

        # Spatial transformation gate: projects full predicted proprioception into block modulation
        if self.spatial_modulation:
            self._spatial_in = nn.Sequential(
                nn.Linear(self.proprio_dim, self.hidden, bias=True),
                nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
                act_fn(),
            )
            self._spatial_transform = BlockLinear(self.hidden, deter, self.blocks)
            self.reset_spatial_parameters()
        self._last_proprio_pred = None

    def reset_spatial_parameters(self):
        """Independently orthogonalize each joint block projection and zero bias."""
        if not self.spatial_modulation:
            return
        with torch.no_grad():
            for b in range(self.blocks):
                nn.init.orthogonal_(self._spatial_transform.weight[:, :, b])
            self._spatial_transform.bias.zero_()

    def _spatial_term(self, proprio_pred):
        spatial_feat = self._spatial_in(proprio_pred)
        return self._spatial_transform(spatial_feat)

    def _candidate(self, reset, cand, proprio_pred):
        if self.spatial_modulation:
            spatial = self._spatial_term(proprio_pred)
            return torch.tanh(spatial * reset * cand)
        else:
            return torch.tanh(reset * cand)

    def forward(self, stoch, deter, action):
        """Deterministic state transition with predicted proprioception modulation."""
        x0, x1, x2 = self._project_inputs(stoch, deter, action)
        proprio_input = torch.cat([x0, x1, x2], dim=-1)
        proprio_pred = self._proprio_pred(proprio_input)
        self._last_proprio_pred = proprio_pred

        reset, cand, update = self._gate_preactivations(deter, x0, x1, x2)
        reset = torch.sigmoid(reset)
        cand = self._candidate(reset, cand, proprio_pred)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim, proprio_dim=26):
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
        self._proprio_dim = int(getattr(config, "proprio_dim", proprio_dim))
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
        kwargs = dict(
            deter=self._deter,
            stoch=self.flat_stoch,
            act_dim=act_dim,
            hidden=self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
        )
        if self._recurrent == "sru":
            kwargs["proprio_dim"] = self._proprio_dim
            kwargs["spatial_modulation"] = getattr(config, "spatial_modulation", True)
        self._deter_net = recurrent_cores[self._recurrent](**kwargs)

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
        if isinstance(self._deter_net, SRUDeter):
            self._deter_net.reset_spatial_parameters()

    @property
    def last_proprio_preds(self):
        return getattr(self, "_last_proprio_preds", None)

    def initial(self, batch_size):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        self._last_proprio_preds = None
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations."""
        # (B, T, E), (B, T, A), ((B, S, K), (B, D)) (B, T)
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        proprio_preds = []
        is_sru = isinstance(self._deter_net, SRUDeter)
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
            if is_sru:
                proprio_preds.append(self._deter_net._last_proprio_pred)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        if is_sru:
            self._last_proprio_preds = torch.stack(proprio_preds, dim=1)
        else:
            self._last_proprio_preds = None
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
