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
    """Block-GRU with an observation-aligned, affine candidate multiplier.

    Spatial input: [flatten(z_{t-1}), bounded(a_{t-2})]. Ordinary dynamics
    retain a_{t-1}. No sensory predictor or auxiliary loss is used.
    """

    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU", spatial_modulation=True):
        super().__init__(deter, stoch, act_dim, hidden, blocks, dynlayers, act)
        self.deter = int(deter)
        self.act_dim = int(act_dim)
        self.spatial_modulation = bool(spatial_modulation)
        self._last_spatial = None
        if self.spatial_modulation:
            self._spatial_transform = nn.Linear(stoch + act_dim, deter, bias=True)
            self.reset_spatial_parameters()

    def reset_spatial_parameters(self):
        if not self.spatial_modulation:
            return
        with torch.no_grad():
            self._spatial_transform.weight.zero_()
            self._spatial_transform.bias.fill_(1.0)

    def _candidate(self, reset, cand, stoch, preceding_action):
        if self.spatial_modulation:
            preceding_action = preceding_action / torch.clip(torch.abs(preceding_action), min=1.0).detach()
            spatial_input = torch.cat([stoch.flatten(1), preceding_action], dim=-1)
            spatial = self._spatial_transform(spatial_input)
            self._last_spatial = spatial.detach()
            return torch.tanh(spatial * reset * cand)
        else:
            self._last_spatial = None
            return torch.tanh(reset * cand)

    def forward(self, stoch, deter, action, preceding_action=None):
        """Candidate modulation; action always drives the forward dynamics."""
        if preceding_action is None:
            raise ValueError("Observation-aligned SRU requires the stochastic state's incoming action")
        x0, x1, x2 = self._project_inputs(stoch, deter, action)
        reset, cand, update = self._gate_preactivations(deter, x0, x1, x2)
        reset = torch.sigmoid(reset)
        cand = self._candidate(reset, cand, stoch, preceding_action)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class SRUDecodeDeter(SRUDeter):
    """SRU candidate conditioned on detached, reconstructed sensory inputs.

    Decoder and camera CNN are supplied by Dreamer at each call; they are not
    registered here, copied, or added to the RSSM optimizer/checkpoint tree.
    """

    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU",
                 spatial_modulation=True, proprio_dim=26, visual_dim=1024):
        super().__init__(deter, stoch, act_dim, hidden, blocks, dynlayers, act, spatial_modulation)
        self.proprio_dim = int(proprio_dim)
        self.visual_dim = int(visual_dim)
        if self.spatial_modulation:
            self._spatial_transform = nn.Linear(self.visual_dim + self.proprio_dim + act_dim, deter, bias=True)
            self.reset_spatial_parameters()

    def forward(self, stoch, deter, action, preceding_action=None, *, sensory_modules=None):
        if preceding_action is None:
            raise ValueError("sru_decode requires the stochastic state's incoming action")
        x0, x1, x2 = self._project_inputs(stoch, deter, action)
        reset, cand, update = self._gate_preactivations(deter, x0, x1, x2)
        if self.spatial_modulation:
            if sensory_modules is None:
                raise ValueError("sru_decode requires Dreamer's decoder and camera CNN")
            decoder, camera_cnn = sensory_modules
            # FP32 isolates this no-grad use from AMP's cached parameter casts.
            # The ordinary reconstruction/encoding passes must still get grads.
            with torch.no_grad(), torch.autocast(device_type=stoch.device.type, enabled=False):
                decoded = decoder(stoch.float(), deter.float())
                image = decoded["image"].mode()
                proprio = decoded["proprioception"].mode()
                # Decoder pixels already use Dreamer's normalized image scale.
                # Do not quantize or divide by 255 a second time.
                visual = camera_cnn(image)
            if visual.shape[-1] != self.visual_dim or proprio.shape[-1] != self.proprio_dim:
                raise ValueError("sru_decode sensory dimensions do not match decoder/CNN outputs")
            preceding_action = preceding_action / torch.clip(torch.abs(preceding_action), min=1.0).detach()
            spatial = self._spatial_transform(torch.cat([visual, proprio, preceding_action], dim=-1))
            self._last_spatial = spatial.detach()
            cand = spatial * cand
        else:
            self._last_spatial = None
        cand = torch.tanh(torch.sigmoid(reset) * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class SRUAuxDeter(SRUDeter):
    """Current-state sensory reconstruction with detached candidate conditioning.

    The extractor receives the input state's incoming action, not the forward
    dynamics action. Auxiliary supervision trains shared state/action projections.
    Recurrent losses train the spatial affine map, but cannot reach the
    extractor through its detached visual/proprioceptive outputs.
    """

    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU",
                 spatial_modulation=True, proprio_dim=26, visual_dim=1024):
        super().__init__(deter, stoch, act_dim, hidden, blocks, dynlayers, act, spatial_modulation)
        self.proprio_dim = int(proprio_dim)
        self.visual_dim = int(visual_dim)
        act_fn = getattr(nn, act)
        self._sensory_extractor = nn.Sequential(
            nn.Linear(3 * self.hidden, self.hidden),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32), act_fn(),
            nn.Linear(self.hidden, self.hidden),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32), act_fn(),
            nn.Linear(self.hidden, self.hidden),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32), act_fn(),
            nn.Linear(self.hidden, self.visual_dim + self.proprio_dim),
        )
        if self.spatial_modulation:
            self._spatial_transform = nn.Linear(self.visual_dim + self.proprio_dim + act_dim, deter)
            self.reset_spatial_parameters()

    def _sensory_predictions(self, x0, x1, incoming_action):
        incoming_action = incoming_action / torch.clip(torch.abs(incoming_action), min=1.0).detach()
        incoming = self._dyn_in2(incoming_action)
        predictions = self._sensory_extractor(torch.cat([x0, x1, incoming], dim=-1))
        return predictions.split([self.visual_dim, self.proprio_dim], dim=-1)

    def predict_sensory(self, stoch, deter, incoming_action):
        """Read out state t using its incoming a_{t-1}, including B,T batches."""
        if incoming_action.shape != (*deter.shape[:-1], self.act_dim):
            raise ValueError("Sensory readout requires the input state's incoming action")
        stoch = stoch.reshape(*deter.shape[:-1], -1)
        return self._sensory_predictions(self._dyn_in0(deter), self._dyn_in1(stoch), incoming_action)

    def forward(self, stoch, deter, action, preceding_action=None):
        if preceding_action is None:
            raise ValueError("Observation-aligned SRU requires the stochastic state's incoming action")
        x0, x1, x2 = self._project_inputs(stoch, deter, action)
        reset, cand, update = self._gate_preactivations(deter, x0, x1, x2)
        if self.spatial_modulation:
            visual, proprio = self._sensory_predictions(x0, x1, preceding_action)
            preceding_action = preceding_action / torch.clip(torch.abs(preceding_action), min=1.0).detach()
            spatial = self._spatial_transform(
                torch.cat([visual.detach(), proprio.detach(), preceding_action], dim=-1)
            )
            self._last_spatial = spatial.detach()
            cand = spatial * cand
        else:
            self._last_spatial = None
        cand = torch.tanh(torch.sigmoid(reset) * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class MotionSRUDeter(Deter):
    """Block-GRU whose candidate is modulated by predicted ego-motion.

    Ground-truth motion is intentionally absent from this module. Both replay
    transitions and imagined transitions use the same differentiable path:
    (deter, stoch, action) -> predicted motion -> spatial multiplier -> deter.
    """

    motion_dim = 6

    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super().__init__(deter, stoch, act_dim, hidden, blocks, dynlayers, act)
        self.deter = int(deter)
        self.use_visual_spatial = False
        act_fn = getattr(torch.nn, act)
        self._motion_pred = nn.Sequential(
            nn.Linear(3 * self.hidden, self.hidden, bias=True),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
            act_fn(),
            nn.Linear(self.hidden, self.hidden, bias=True),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
            act_fn(),
            nn.Linear(self.hidden, self.motion_dim, bias=True),
        )
        # Deliberately linear: predicted motion -> feature-wise multiplier.
        self._spatial_transform = nn.Linear(self.motion_dim, self.deter, bias=True)
        self.reset_spatial_parameters()
        self._last_motion_pred = None
        self._last_spatial = None

    def reset_spatial_parameters(self):
        """Start exactly at the vanilla block-GRU candidate."""
        with torch.no_grad():
            nn.init.zeros_(self._spatial_transform.weight)
            nn.init.ones_(self._spatial_transform.bias)

    def forward(self, stoch, deter, action):
        x0, x1, x2 = self._project_inputs(stoch, deter, action)
        motion_pred = self._motion_pred(torch.cat([x0, x1, x2], dim=-1))
        spatial = self._spatial_transform(motion_pred)

        # Keep the prediction attached for its supervised loss. The cached
        # spatial tensor is diagnostics-only; the local tensor below remains
        # attached to the recurrent computation.
        self._last_motion_pred = motion_pred
        self._last_spatial = spatial.detach()

        reset, cand, update = self._gate_preactivations(deter, x0, x1, x2)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(spatial * reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class MotionVisualSRUDeter(MotionSRUDeter):
    """Motion-SRU extended with a causal predicted camera-CNN feature."""

    def __init__(
        self,
        deter,
        stoch,
        act_dim,
        hidden,
        blocks,
        dynlayers,
        act="SiLU",
        visual_dim=1024,
    ):
        super().__init__(deter, stoch, act_dim, hidden, blocks, dynlayers, act)
        self.use_visual_spatial = True
        self.visual_dim = int(visual_dim)
        act_fn = getattr(torch.nn, act)
        self._visual_pred = nn.Sequential(
            nn.Linear(3 * self.hidden, self.hidden, bias=True),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
            act_fn(),
            nn.Linear(self.hidden, self.hidden, bias=True),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
            act_fn(),
            nn.Linear(self.hidden, self.visual_dim, bias=True),
        )
        self._spatial_transform = nn.Linear(
            self.motion_dim + self.visual_dim, self.deter, bias=True
        )
        self.reset_spatial_parameters()
        self._last_visual_pred = None
        self._last_spatial_input = None

    def forward(self, stoch, deter, action):
        x0, x1, x2 = self._project_inputs(stoch, deter, action)
        predictor_input = torch.cat([x0, x1, x2], dim=-1)
        motion_pred = self._motion_pred(predictor_input)
        visual_pred = self._visual_pred(predictor_input)
        spatial_input = torch.cat([motion_pred, visual_pred], dim=-1)
        spatial = self._spatial_transform(spatial_input)

        # Ground-truth motion and actual CNN features never enter this path.
        self._last_motion_pred = motion_pred
        self._last_visual_pred = visual_pred
        self._last_spatial_input = spatial_input
        self._last_spatial = spatial.detach()

        reset, cand, update = self._gate_preactivations(deter, x0, x1, x2)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(spatial * reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class ActionVisualSRUDeter(Deter):
    """Candidate SRU driven by the supplied action and predicted camera feature.

    Predict v_t from (h_{t-1}, z_{t-1}, a_{t-1}), then form the multiplier
    from [a_{t-1}, predicted v_t]. No current observation or motion target is
    read by default, so posterior updates and imagination share the same path.
    Evaluation observation updates may explicitly substitute current CNN features;
    imagination always retains the prediction-only path.
    """

    def __init__(
        self, deter, stoch, act_dim, hidden, blocks, dynlayers,
        act="SiLU", visual_dim=1024,
    ):
        super().__init__(deter, stoch, act_dim, hidden, blocks, dynlayers, act)
        self.deter = int(deter)
        self.act_dim = int(act_dim)
        self.visual_dim = int(visual_dim)
        self.use_visual_spatial = True
        act_fn = getattr(torch.nn, act)
        self._visual_pred = nn.Sequential(
            nn.Linear(3 * self.hidden, self.hidden, bias=True),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
            act_fn(),
            nn.Linear(self.hidden, self.hidden, bias=True),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
            act_fn(),
            nn.Linear(self.hidden, self.hidden, bias=True),
            nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32),
            act_fn(),
            nn.Linear(self.hidden, self.visual_dim, bias=True),
        )
        self._spatial_transform = nn.Linear(
            self.act_dim + self.visual_dim, self.deter, bias=True
        )
        self.reset_spatial_parameters()
        self._last_visual_pred = None
        self._last_spatial_input = None
        self._last_spatial = None

    def reset_spatial_parameters(self):
        with torch.no_grad():
            nn.init.zeros_(self._spatial_transform.weight)
            nn.init.ones_(self._spatial_transform.bias)

    def forward(self, stoch, deter, action, visual_features=None):
        x0, x1, x2 = self._project_inputs(stoch, deter, action)
        visual_pred = self._visual_pred(torch.cat([x0, x1, x2], dim=-1))
        if visual_features is not None:
            if self.training:
                raise ValueError("Observed SRU visual features are evaluation-only")
            if visual_features.shape != visual_pred.shape:
                raise ValueError("Observed SRU features must match the predicted CNN feature shape")
        # Use the actual supplied action, not a predicted motion or embedding.
        # _project_inputs retains the existing bounded-action predictor path.
        spatial_visual = visual_pred if visual_features is None else visual_features
        spatial_input = torch.cat([action, spatial_visual], dim=-1)
        spatial = self._spatial_transform(spatial_input)
        self._last_visual_pred = visual_pred
        self._last_spatial_input = spatial_input
        self._last_spatial = spatial.detach()

        reset, cand, update = self._gate_preactivations(deter, x0, x1, x2)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(spatial * reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim, visual_dim=1024, proprio_dim=26):
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
        self.requires_action_history = self._recurrent in {"sru", "sru_decode", "sru_aux"}
        self.requires_sensory_decode = self._recurrent == "sru_decode"
        self.requires_sensory_aux = self._recurrent == "sru_aux"
        self._use_motion_sru = bool(getattr(config, "use_motion_sru", False))
        self._use_visual_spatial = bool(
            getattr(config, "use_visual_spatial", False)
        )
        recurrent_cores = {
            "gru": Deter,
            "sru": SRUDeter,
            "sru_decode": SRUDecodeDeter,
            "sru_aux": SRUAuxDeter,
            "action_visual_sru": ActionVisualSRUDeter,
        }
        if self._recurrent not in recurrent_cores:
            choices = ", ".join(recurrent_cores)
            raise ValueError(f"rssm.recurrent must be one of {{{choices}}}, got {self._recurrent!r}")
        if self._recurrent == "action_visual_sru" and (
            self._use_motion_sru or self._use_visual_spatial
        ):
            raise ValueError("action_visual_sru requires legacy SRU flags to be false")
        if self._use_motion_sru and self._recurrent != "gru":
            raise ValueError(
                "rssm.use_motion_sru=true requires rssm.recurrent='gru'; "
                "the observation-aligned SRU and motion SRU are separate variants"
            )
        if self._use_visual_spatial and not self._use_motion_sru:
            raise ValueError(
                "rssm.use_visual_spatial=true requires "
                "rssm.use_motion_sru=true"
            )
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
        if self.requires_action_history:
            kwargs["spatial_modulation"] = getattr(config, "spatial_modulation", True)
        if self.requires_sensory_decode or self.requires_sensory_aux:
            kwargs["proprio_dim"] = int(proprio_dim)
            kwargs["visual_dim"] = int(visual_dim)
        if self._use_visual_spatial or self._recurrent == "action_visual_sru":
            kwargs["visual_dim"] = int(visual_dim)
        if self._use_visual_spatial:
            deter_core = MotionVisualSRUDeter
        elif self._use_motion_sru:
            deter_core = MotionSRUDeter
        else:
            deter_core = recurrent_cores[self._recurrent]
        self._deter_net = deter_core(**kwargs)

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
        if isinstance(self._deter_net, (SRUDeter, MotionSRUDeter, ActionVisualSRUDeter)):
            self._deter_net.reset_spatial_parameters()
        self._last_spatial_terms = None
        self._last_motion_preds = None
        self._last_visual_preds = None

    @property
    def last_spatial_terms(self):
        return self._last_spatial_terms

    def predict_sensory(self, stoch, deter, incoming_action):
        if not self.requires_sensory_aux:
            raise ValueError("Sensory auxiliary reconstruction requires recurrent='sru_aux'")
        return self._deter_net.predict_sensory(stoch, deter, incoming_action)

    @property
    def last_motion_preds(self):
        return self._last_motion_preds

    @property
    def last_visual_preds(self):
        return self._last_visual_preds

    def initial(self, batch_size):
        """Initial (stoch, deter), plus zero incoming_action for aligned SRU."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        self._last_motion_preds = None
        self._last_visual_preds = None
        self._last_spatial_terms = None
        if self.requires_action_history:
            incoming_action = torch.zeros(batch_size, self._act_dim, dtype=torch.float32, device=self._device)
            return stoch, deter, incoming_action
        return stoch, deter

    def observe(self, embed, action, initial, reset, *, sensory_modules=None):
        """Posterior rollout: action[:, t] is a_{t-1}, not outgoing a_t.

        Aligned initial state is (z_{k-1}, h_{k-1}, a_{k-2}). Subsequent
        incoming actions are carried locally; resets mask both action inputs.
        """
        # (B, T, E), (B, T, A), ((B, S, K), (B, D)) (B, T)
        L = action.shape[1]
        if self.requires_action_history:
            if len(initial) != 3:
                raise ValueError("Observation-aligned SRU initial state requires incoming_action; no replay fallback")
            stoch, deter, incoming_action = initial
        else:
            stoch, deter = initial
        stochs, deters, logits = [], [], []
        motion_preds = []
        visual_preds = []
        spatial_terms = []
        is_motion_sru = isinstance(self._deter_net, MotionSRUDeter)
        is_action_visual_sru = isinstance(self._deter_net, ActionVisualSRUDeter)
        track_visual = is_action_visual_sru or (is_motion_sru and self._deter_net.use_visual_spatial)
        track_spatial = is_action_visual_sru or is_motion_sru or (
            self.requires_action_history and self._deter_net.spatial_modulation
        )
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            if self.requires_action_history:
                stoch, deter, logit, incoming_action = self.obs_step(
                    stoch, deter, action[:, i], embed[:, i], reset[:, i], preceding_action=incoming_action,
                    sensory_modules=sensory_modules,
                )
            else:
                stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
            if is_motion_sru:
                motion_preds.append(self._deter_net._last_motion_pred)
            if track_visual:
                visual_preds.append(self._deter_net._last_visual_pred)
            if track_spatial:
                spatial_terms.append(self._deter_net._last_spatial)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        if is_motion_sru:
            self._last_motion_preds = torch.stack(motion_preds, dim=1)
        else:
            self._last_motion_preds = None
        if track_visual:
            self._last_visual_preds = torch.stack(visual_preds, dim=1)
        else:
            self._last_visual_preds = None
        self._last_spatial_terms = torch.stack(spatial_terms, dim=1) if track_spatial else None
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset, visual_features=None, *, preceding_action=None, sensory_modules=None):
        """Construct (z_t, h_t) from z_{t-1}, h_{t-1}, a_{t-1}.

        Aligned mode additionally takes a_{t-2} and returns the masked
        a_{t-1} as a fourth result, the incoming action associated with z_t.
        """
        # (B, S, K), (B, D), (B, A), (B, E), (B,)
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

        # Deterministic transition then posterior logits conditioned on embed.
        # (B, D)
        if self.requires_action_history:
            if preceding_action is None:
                raise ValueError("Observation-aligned SRU requires incoming_action in posterior state")
            if visual_features is not None:
                raise ValueError("Observation-aligned SRU does not accept observed visual features")
            preceding_action = torch.where(rpad(reset, preceding_action.dim() - reset.dim()),
                                          torch.zeros_like(preceding_action), preceding_action)
            if self.requires_sensory_decode:
                deter = self._deter_net(stoch, deter, prev_action, preceding_action=preceding_action,
                                       sensory_modules=sensory_modules)
            else:
                deter = self._deter_net(stoch, deter, prev_action, preceding_action=preceding_action)
        elif visual_features is None:
            deter = self._deter_net(stoch, deter, prev_action)
        else:
            if not isinstance(self._deter_net, ActionVisualSRUDeter):
                raise ValueError("Observed SRU features require action_visual_sru")
            deter = self._deter_net(stoch, deter, prev_action, visual_features=visual_features)
        # (B, D + E)
        x = torch.cat([deter, embed], dim=-1)
        # (B, S, K)
        logit = self._obs_net(x)

        # Sample discrete stochastic state via straight-through Gumbel-Softmax.
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        if self.requires_action_history:
            return stoch, deter, logit, prev_action
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action, *, preceding_action=None, sensory_modules=None):
        """Prior transition; aligned mode returns incoming action as result 3."""

        # (B, D)
        if self.requires_action_history:
            if preceding_action is None:
                raise ValueError("Observation-aligned SRU requires incoming_action in prior state")
            if self.requires_sensory_decode:
                deter = self._deter_net(stoch, deter, prev_action, preceding_action=preceding_action,
                                       sensory_modules=sensory_modules)
            else:
                deter = self._deter_net(stoch, deter, prev_action, preceding_action=preceding_action)
        else:
            deter = self._deter_net(stoch, deter, prev_action)
        # (B, S, K)
        stoch, _ = self.prior(deter)
        if self.requires_action_history:
            return stoch, deter, prev_action
        return stoch, deter

    def prior(self, deter):
        """Compute prior distribution parameters and sample stoch."""

        # (B, S, K)
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions, *, preceding_action=None, sensory_modules=None):
        """Roll out prior dynamics given a sequence of actions."""
        # (B, S, K), (B, D), (B, T, A)
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            if self.requires_action_history:
                stoch, deter, preceding_action = self.img_step(
                    stoch, deter, actions[:, i], preceding_action=preceding_action, sensory_modules=sensory_modules,
                )
            else:
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
