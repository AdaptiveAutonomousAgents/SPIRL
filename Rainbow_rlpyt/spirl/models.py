import torch
import torch.nn.functional as F
import torch.nn as nn

from rlpyt.models.utils import scale_grad
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from spirl.rainbow_utils import count_parameters
from .keypoints import HEIGHT, WEIGHT, NUM_RAM_KPS
import numpy as np
from timm.models.layers import Mlp, DropPath
# from timm.models.vision_transformer import Block
from timm.models.vision_transformer import trunc_normal_
from functools import partial
import pdb


class SPRCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            image_shape,  # env_spaces.observation.shape
            output_size,  # env_spaces.action.n
            n_atoms,
            dueling,
            noisy_nets,
            classifier,
            imagesize,
            distributional,
            dqn_hidden_size,
            renormalize,
            dropout,
            noisy_nets_std,
            encoder_type,
            obs_type,
            obs_args,
            game,
            dict_input,
            attention_args,
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.noisy = noisy_nets
        self.classifier_type = classifier

        self.distributional = distributional
        n_atoms = 1 if not self.distributional else n_atoms
        self.dqn_hidden_size = dqn_hidden_size

        self.transforms = []
        self.eval_transforms = []

        self.dueling = dueling

        self.image_obs = (obs_type == 'image')
        self.ram_obs = (obs_type == 'ram')
        self.ram_kp_pos_obs = (obs_type == 'ramKpPos')  # extract keypoint position, without velocity
        self.ram_kp_info_obs = (obs_type == 'ramKpInfo')  # extract keypoint position, w/ other information
        self.vit_rec_obs = (obs_type == 'vitRec')  # select keypoint patches based on reconstruction error
        self.vit_ram_obs = (obs_type == 'vitRam')  # select keypoint patches based on ram information
        self.obs_args = obs_args
        self.image_shape = image_shape
        self.dict_input = dict_input  # the last row is len_keep
        if self.image_obs:
            f, c = image_shape[:2]  # frame_stack, channel
            in_channels = np.prod(image_shape[:2])
            fake_input = torch.zeros(1, f*c, imagesize, imagesize)
            feature_args = None
        elif self.ram_obs or self.ram_kp_pos_obs:
            # NOTE: in fact, here the image_shape refer to obs_shape
            f = image_shape[0]  # number of frames stacked
            input_dims = np.prod(image_shape)  # f * ram size 128
            fake_input = torch.zeros(1, input_dims)  # fake batch size = 1
            feature_args = None
        elif self.vit_rec_obs or self.vit_ram_obs:
            feature_type = obs_args['features_extractor_kwargs']['feature']
            f = image_shape[0]  # number of frames stacked
            if feature_type in ['all', 'x', 'xCls']:
                assert encoder_type == "AttentionModel"
                # image_shape (f, #pathes+1, embed_dim)
                fake_input = torch.zeros(1, f, image_shape[-2], image_shape[-1])
                if self.dict_input:
                    fake_input[:, :, -1, :] = 1.
            else:
                input_dims = np.prod(image_shape)  # f * ram size 128
                fake_input = torch.zeros(1, input_dims)  # fake batch size = 1
                need_norm = ('norm' in feature_type)
                feature_args = {
                    'add_pos_type': obs_args['features_extractor_kwargs']['add_pos_type'],
                    'embed_dim': obs_args['features_extractor_kwargs']['embed_dim'],
                    'need_norm': need_norm,
                }
        elif self.ram_kp_info_obs:
            f = image_shape[0]
            self.info_size = (f-1)*NUM_RAM_KPS[game]*2 + NUM_RAM_KPS[game]*(NUM_RAM_KPS[game]-1)*f
            input_dims = np.prod((image_shape[0], image_shape[1])) + self.info_size
            fake_input = torch.zeros(1, input_dims)  # fake batch size = 1
            feature_args = None
        else:
            raise NotImplementedError

        if encoder_type == "ConvCanonical":
            self.encoder = Conv2dModel(
                in_channels=in_channels,
                channels=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                paddings=[0, 0, 0],
                use_maxpool=False,
                dropout=dropout,
            )
        elif encoder_type == "ConvDataEfficient":
            self.encoder = Conv2dModel(
                in_channels=in_channels,
                channels=[32, 64],
                kernel_sizes=[5, 5],
                strides=[5, 5],
                paddings=[3, 1],
                use_maxpool=False,
                dropout=dropout,
            )
        elif encoder_type == "MLPW0D1":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[128], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW0D2":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[128, 128], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW0D3":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[128, 128, 128], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW0D4":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[128, 128, 128, 128], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW1D1":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[256], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW2D1":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[512], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW1D2":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[256, 256], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW2D2":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[512, 256], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW3D2":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[512, 512], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW1D3":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[256, 256, 256], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "MLPW2D3":
            self.encoder = MLPModel(
                input_dim=input_dims,
                mlp_hidden_dims=[512, 256, 256], 
                dropout=dropout,
                feature_args=feature_args,
            )
        elif encoder_type == "AttentionModel":
            self.encoder = AttentionModel(
                embed_dim=image_shape[-1],
                num_frame=image_shape[0],
                dict_input=self.dict_input,
                **attention_args,
            )
        else:
            raise NotImplementedError

        fake_output = self.encoder(fake_input)
        # data_dim: len(input shape) for encoder
        # latent_dim: len(output shape) for encoder
        if self.image_obs:
            self.hidden_size = fake_output.shape[-3]  # fake_output.shape: (batch_size, last channel, spatial_feature_h, spatial_feature_w)
            self.pixels = fake_output.shape[-1] * fake_output.shape[-2]  # size of the feature map
            self.data_dim = self.latent_dim = 3  # len((f*c, h, w))
        elif self.ram_obs or self.ram_kp_pos_obs or self.ram_kp_info_obs:
            self.hidden_size = 1  # fake_output.shape: (batch_size, the last mlp_hidden_dim)
            self.pixels = fake_output.shape[-1]  # the last mlp hidden size 
            self.data_dim = self.latent_dim = 1  # len((f * ram_size,))
        elif self.vit_rec_obs or self.vit_ram_obs:
            self.hidden_size = 1  # fake_output.shape: (batch_size, the last mlp_hidden_dim)
            self.pixels = fake_output.shape[-1]  # the last mlp hidden size 
            if encoder_type == "AttentionModel":
                self.data_dim = 3  # len((f, len_keep, embed_dim))
                self.latent_dim = 1  # len((latent_dim))
            else:
                self.data_dim = self.latent_dim = 1  # len((latent))
        else:
            raise NotImplementedError
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        self.num_actions = output_size

        if dueling:
            self.head = DQNDistributionalDuelingHeadModel(self.hidden_size,  # input_channles
                                                          output_size,
                                                          hidden_size=self.dqn_hidden_size,
                                                          pixels=self.pixels,  # the input size for the 1st layer will be input_channels*pixels
                                                          noisy=self.noisy,
                                                          n_atoms=n_atoms,
                                                          std_init=noisy_nets_std,
                                                          image_obs=self.image_obs,
                                                          )
        else:
            self.head = DQNDistributionalHeadModel(self.hidden_size,
                                                   output_size,
                                                   hidden_size=self.dqn_hidden_size,
                                                   pixels=self.pixels,
                                                   noisy=self.noisy,
                                                   n_atoms=n_atoms,
                                                   std_init=noisy_nets_std,
                                                   image_obs=self.image_obs,
                                                   )

        self.renormalize = renormalize

        print("Initialized model with {} parameters".format(count_parameters(self)))

    def set_sampling(self, sampling):
        if self.noisy:
            self.head.set_sampling(sampling)

    @torch.no_grad()
    def transform(self, obs, augment=False):
        # image shape: ((T,B), f*c, h, w) if image_obs, ((T,B), f*ram_size) if ram_obs
        # TODO: modify this part for namedarrayTuple obs
        obs = obs.float() if obs.dtype == torch.uint8 else obs
        if self.ram_kp_info_obs:
            f, num_pos = obs.shape[-2], obs.shape[-1]
            num_kps = num_pos//2
            info = torch.zeros(list(obs.shape[:-2])+[self.info_size], dtype=obs.dtype, device=obs.device)
            idx = 0
            # NOTE: delt will have negative values
            # delt_x, delt_y for each object in consecutive frames
            for id_f in range(f-1):
                info[..., idx: idx+num_pos] = obs[...,id_f,:] - obs[...,id_f+1,:]
                idx += num_pos
            # delt_x, delt_y for objects in one frame
            for ip1 in range(num_kps-1):
                for ip2 in range(ip1+1, num_kps):
                    info[..., idx: idx+2*f] = (obs[...,:,ip1*2:ip1*2+2] - obs[...,:,ip2*2:ip2*2+2]).flatten(-2, -1)
                    idx += 2*f
            obs = obs.flatten(-2, -1)
            obs = torch.cat([obs, info], dim=-1)
            if self.obs_args['raw_ram_input']:
                obs /= 255.
            else:
                obs[..., ::2] /= (1.0*WEIGHT)
                obs[..., 1::2] /= (1.0*HEIGHT)
        elif self.ram_kp_pos_obs:
            obs = obs.flatten(-2, -1)  # (T, B), f, len(ram) --> (T, B), f*len(ram)
            if self.obs_args['raw_ram_input']:
                obs /= 255.
            else:
                obs[..., ::2] /= (1.0*WEIGHT)
                obs[..., 1::2] /= (1.0*HEIGHT)
        elif self.image_obs:
            obs = obs.flatten(-4, -3)  # (T, B), f, c, h, w --> (T, B), f*c, h, w
            obs /= 255.
        elif self.vit_rec_obs or self.vit_ram_obs:
            if type(self.encoder) != AttentionModel:
                obs = obs.flatten(-2, -1)  # f * embed_dim
        else:
            raise NotImplementedError

        return obs

    def stem_parameters(self):
        return list(self.encoder.parameters()) + list(self.head.parameters())

    def stem_forward(self, img, prev_action=None, prev_reward=None):
        """Returns the normalized output of encoder (conv or mlp)."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, self.data_dim)  # data: (f*c, h, w) or (f * ram_len)

        enc_out = self.encoder(img.view(T * B, *img_shape))  # Fold if T dimension.
        if self.renormalize:
            enc_out = renormalize(enc_out, -self.data_dim)
        return enc_out

    def head_forward(self,
                     enc_out,
                     prev_action,
                     prev_reward,
                     logits=False):
        lead_dim, T, B, img_shape = infer_leading_dims(enc_out, self.latent_dim)
        p = self.head(enc_out)

        if self.distributional:
            if logits:
                p = F.log_softmax(p, dim=-1)
            else:
                p = F.softmax(p, dim=-1)
        else:
            p = p.squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    def forward(self, observation,
                prev_action, prev_reward,
                train=False, eval=False):
        """
        For convenience reasons with DistributedDataParallel the forward method
        has been split into two cases, one for training and one for eval.
        """
        if train:
            # case 1: algos.loss --> agent.__call__ --> agent.model (HERE)
            # observation.shape: e.g. ram_obs: (n-steps+1, batch_size, f, ram_size)
            log_pred_ps = []
            pred_reward = []
            pred_latents = []
            # input_obs = self.transform(input_obs, augment=True)  # original line
            input_obs = self.transform(observation[0])  # dqn train only on the first step
            latent = self.stem_forward(input_obs,
                                       prev_action[0],
                                       prev_reward[0])
            log_pred_ps.append(self.head_forward(latent,
                                                 prev_action[0],
                                                 prev_reward[0],
                                                 logits=True))
            pred_latents.append(latent)
            return log_pred_ps

        else:
            # case 1: evalCollector --> agent.run --> network.select_action --> HERE
            #         In this case, obs has been processed in agent.run
            # case 2: algos.loss --> self.rl_loss --> agent.__call__ --> HERE
            #         In this case, obs is the same as stored in replay_buffer (i.e. uint8 data)
            obs = self.transform(observation)

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, obs_shape = infer_leading_dims(obs, self.data_dim)

            enc_out = self.encoder(obs.view(T * B, *obs_shape))  # Fold if T dimension.
            if self.renormalize:  # layer-wise scale to [0, 1]
                enc_out = renormalize(enc_out, -self.data_dim)
            p = self.head(enc_out)

            if self.distributional:
                p = F.softmax(p, dim=-1)
            else:
                p = p.squeeze(-1)

            p = p.view(observation.shape[0],  # batch size
                       *p.shape[1:])  # #actions, #atoms

            # Restore leading dimensions: [T,B], [B], or [], as input.
            p = restore_leading_dims(p, lead_dim, T, B)
            return p

    def select_action(self, obs):
        # image obs. shape: (T, B), f, c, h, w, ram obs.shape: (T, B), f, ram_size
        value = self.forward(obs, None, None, train=False, eval=True)

        if self.distributional:
            value = from_categorical(value, logits=False, limit=10)
        return value

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        return next_state, reward_logits


class MLPHead(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=-1,
                 pixels=30,
                 noisy=0):
        super().__init__()
        if noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        self.noisy = noisy
        if hidden_size <= 0:
            hidden_size = input_channels*pixels
        self.linears = [linear(input_channels*pixels, hidden_size),
                        linear(hidden_size, output_size)]
        layers = [nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        if not noisy:
            self.network.apply(weights_init)
        self._output_size = output_size

    def forward(self, input):
        return self.network(input)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class DQNDistributionalHeadModel(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=256,
                 pixels=30,
                 n_atoms=51,
                 noisy=0,
                 std_init=0.1,
                 image_obs=True,
                 ):
        super().__init__()
        if noisy:
            linear = NoisyLinear
            self.linears = [linear(input_channels*pixels, hidden_size, std_init=std_init),
                            linear(hidden_size, output_size * n_atoms, std_init=std_init)]
        else:
            linear = nn.Linear
            self.linears = [linear(input_channels*pixels, hidden_size),
                            linear(hidden_size, output_size * n_atoms)]
        if image_obs:
            layers = [nn.Flatten(-3, -1)]
        else:
            layers = []
        layers.extend([self.linears[0],
                       nn.ReLU(),
                       self.linears[1]])
        self.network = nn.Sequential(*layers)
        if not noisy:  # since for noisy layer, they declare std_init
            self.network.apply(weights_init)
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.network(input).view(-1, self._output_size, self._n_atoms)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class DQNDistributionalDuelingHeadModel(torch.nn.Module):
    """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 pixels=30,
                 n_atoms=51,
                 hidden_size=256,
                 grad_scale=2 ** (-1 / 2),
                 noisy=0,
                 std_init=0.1,
                 image_obs=True,
                 ):
        super().__init__()
        if noisy:
            self.linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
                            NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, n_atoms, std_init=std_init)
                            ]
        else:
            self.linears = [nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, output_size * n_atoms),
                            nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, n_atoms)
                            ]

        self.advantage_layers = []
        self.value_layers = []
        if image_obs:
            # flatten c h w dimensions
            self.advantage_layers.append(nn.Flatten(-3, -1))
            self.value_layers.append(nn.Flatten(-3, -1))

        self.advantage_layers.extend([self.linears[0],
                                      nn.ReLU(),
                                      self.linears[1]])
        self.value_layers.extend([self.linears[2],
                                  nn.ReLU(),
                                  self.linears[3]])
        self.advantage_hidden = nn.Sequential(*self.advantage_layers[:-1])
        self.advantage_out = self.advantage_layers[-1]
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.value = nn.Sequential(*self.value_layers)
        self.network = self.advantage_hidden
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class QL1Head(nn.Module):
    def __init__(self, head, dueling=False, type="noisy advantage"):
        super().__init__()
        self.head = head
        self.noisy = "noisy" in type
        self.dueling = dueling
        self.encoders = nn.ModuleList()
        self.relu = "relu" in type
        value = "value" in type
        advantage = "advantage" in type
        if self.dueling:
            if value:
                self.encoders.append(self.head.value[1])
            if advantage:
                self.encoders.append(self.head.advantage_hidden[1])
        else:
            self.encoders.append(self.head.network[1])

        self.out_features = sum([e.out_features for e in self.encoders])

    def forward(self, x):
        x = x.flatten(-3, -1)
        representations = []
        for encoder in self.encoders:
            encoder.noise_override = self.noisy
            representations.append(encoder(x))
            encoder.noise_override = None
        representation = torch.cat(representations, -1)
        if self.relu:
            representation = F.relu(representation)

        return representation


def weights_init(m):
    if isinstance(m, Conv2dSame):
        torch.nn.init.kaiming_uniform_(m.layer.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.layer.bias)
    elif isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.bias)
    else:
        print(f'******** no special initialization for {m} ********')
        # raise NotImplementedError()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, bias=True):
        super(NoisyLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sampling = True
        self.noise_override = None
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        if not self.bias:
            self.bias_mu.fill_(0)
            self.bias_sigma.fill_(0)
        else:
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # Self.training alone isn't a good-enough check, since we may need to
        # activate .eval() during sampling even when we want to use noise
        # (due to batchnorm, dropout, or similar).
        # The extra "sampling" flag serves to override this behavior and causes
        # noise to be used even when .eval() has been called.
        if self.noise_override is None:
            use_noise = self.training or self.sampling
        else:
            use_noise = self.noise_override
        if use_noise:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


def maybe_transform(image, transform, alt_transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise

class MLPModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        mlp_hidden_dims=[256],  # include the last layer
        nonlinearity=nn.ReLU,
        dropout=0.,
        feature_args=None,
        ):
        super().__init__()
        if feature_args is not None and feature_args['need_norm']:
            self.feature_dim = feature_args['embed_dim'] + \
                (0 if feature_args['add_pos_type'] == 'sinusoid' else 2)
            self.feat_pool_norm = nn.LayerNorm(
                self.feature_dim, eps=1e-6)
        else:
            self.feat_pool_norm = None
        in_dims = [input_dim] + mlp_hidden_dims[:-1]
        mlp_layers = [torch.nn.Linear(in_dim, out_dim) 
            for (in_dim, out_dim) in zip(in_dims, mlp_hidden_dims)]

        sequence = list()  # NOTE: assump that input has been flattened
        for linear_layer in mlp_layers:
            sequence.extend([linear_layer, nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))

        self.mlp = torch.nn.Sequential(*sequence)

    def forward(self, input):
        if self.feat_pool_norm is not None:
            B = input.shape[0]
            input = input.view(-1, self.feature_dim)
            input = self.feat_pool_norm(input)
            input = input.view(B, -1)
        return self.mlp(input)


class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            dropout=0.,
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)


def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "none" or type is None:
        return nn.Identity()


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            residual=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.residual = residual

    def forward(self, x):
        if self.residual:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        else:
            x = self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class AttentionModel(nn.Module):
    def __init__(
        self,
        one_frame_one_cls,
        embed_dim,
        num_frame,
        depth,
        recurrent,
        num_heads,
        proj_embed_dim,  # aviliable if proj_embed_dim>0
        one_frame_one_proj,
        global_pool,
        norm_out,
        rec_thres_pad,
        dict_input=False,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        input_xCls=False,
        rec_thres_div=False,
        residual=True,
    ):
        super().__init__()
        self.one_frame_one_proj = one_frame_one_proj
        self.one_frame_one_cls = one_frame_one_cls
        self.global_pool = global_pool
        self.norm_out = norm_out
        self.dict_input = dict_input
        self.rec_thres_pad = rec_thres_pad
        self.input_xCls = input_xCls
        self.rec_thres_div = rec_thres_div
        self.residual = residual

        if proj_embed_dim > 0:
            if one_frame_one_proj:
                self.proj = [nn.Linear(embed_dim, proj_embed_dim) 
                             for _ in range(num_frame)]
            else:
                self.proj = nn.Linear(embed_dim, proj_embed_dim)
            self.have_proj = True
            self.block_embed_dim = proj_embed_dim
        else:
            self.have_proj = False
            self.block_embed_dim = embed_dim

        if self.input_xCls:
            assert not self.have_proj
            assert not self.dict_input

        # shape for cls_token: (#batch, #frame_stack, #len_keep, embed_dim)
        if one_frame_one_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, num_frame, 1, self.block_embed_dim), requires_grad=True)
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, self.block_embed_dim), requires_grad=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=self.block_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                residual=self.residual)
            for i in range(depth)])
        self.recurrent = recurrent
        self.norm = norm_layer(self.block_embed_dim)
        self.init_weights()  # same as timm.vision_transformer

    def init_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        if self.input_xCls:
            frame_cls = x[:, :, 0, :]
            x = x[:, : , 1:, :]

        if self.dict_input and not self.rec_thres_pad:
            # raise NotImplementedError  # need to solve bus error
            B, F, _, D = x.shape
            x_len_keep = x[..., -1, 0]  # shape (B, F)
            x = x[..., :-1, :]
            if self.have_proj:
                if self.one_frame_one_proj:
                    proj_x = [self.proj[i](x[:, i: i+1, ...]) for i in range(F)]
                    x = torch.cat(proj_x, dim=1)
                else:
                    x = self.proj(x)
                D = x.shape[-1]

            outcome = torch.empty(size=(B, F, D), dtype=x.dtype, device=x.device)
            for i in range(B):
                for j in range(F):
                    L = max(int(x_len_keep[i][j].item()), 1)
                    feat = x[i][j][:L]
                    if self.one_frame_one_cls:
                        cls_expand = self.cls_token[0][j]
                    else:
                        cls_expand = self.cls_token[0][0]
                    feat = torch.cat((cls_expand, feat), dim=-2).reshape(1, L+1, D) # 1, len_keep+1, embed_dim
                    for _ in range(self.recurrent):
                        feat = self.blocks(feat)  # (1, len_keep + 1, embed_dim)
                    if self.global_pool:  # avg
                        feat = feat[0, 1:, :].mean(dim=-2)  # global pool without cls token
                        if self.norm_out:
                            out_feat = self.norm(feat)
                        else:
                            out_feat = feat
                    else:  # cls
                        if self.norm_out:
                            feat = self.norm(feat)
                        out_feat = feat[0, 0:1, :]
                    outcome[i][j] = out_feat
                    if self.rec_thres_div:
                        outcome[i][j] /= L
        else:
            B, F, L, D = x.shape
            # NOTE: if L is not the same for all frames, then we have to calculate it one by one...
            #       otherwise, we can forward x by a batch
            if self.have_proj:
                if self.one_frame_one_proj:
                    proj_x = [self.proj[i](x[:, i: i+1, ...]) for i in range(F)]
                    x = torch.cat(proj_x, dim=1)
                else:
                    x = self.proj(x)
                D = x.shape[-1]

            if self.one_frame_one_cls:
                cls_expand = self.cls_token.expand(B, -1, -1, -1)
            else:
                cls_expand = self.cls_token.expand(B, F, -1, -1)
            x = torch.cat((cls_expand, x), dim=-2).reshape(B*F, L+1, D) # B, F, len_keep+1, embed_dim
            for _ in range(self.recurrent):
                x = self.blocks(x)  # (B, len_keep + 1, embed_dim)
            x = x.reshape(B, F, L+1, D)

            if self.global_pool:
                x = x[:, :, 1:, :].mean(dim=-2)  # global pool without cls token
                if self.norm_out:
                    outcome = self.norm(x)
                else:
                    outcome = x
            else:
                if self.norm_out:
                    x = self.norm(x)
                outcome = x[:, :, 0, :]

        if self.input_xCls:
            outcome = torch.cat((outcome, frame_cls), dim=-1)

        return outcome.reshape(B, -1)


def _init_vit_weights(module: nn.Module, name: str='', head_bias: float=0.):
    # same with mae's initialization
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type="bn"):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            init_normalization(out_channels, norm_type),
            Conv2dSame(out_channels, out_channels, 3),
            init_normalization(out_channels, norm_type),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class Conv2dSame(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                            stride=stride, padding=ka)
        )

    def forward(self, x):
        return self.net(x)


def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit*2+1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def from_categorical(distribution, limit=300, logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    weights = torch.linspace(-limit, limit, num_atoms, device=distribution.device).float()
    return distribution @ weights


class TransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 norm_type="bn",
                 renormalize=True,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize
        self.residual = residual
        layers = [Conv2dSame(channels+num_actions, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size,
                                        norm_type))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])

        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)

        self.network = nn.Sequential(*layers)
        self.reward_predictor = RewardPredictor(channels,
                                                pixels=pixels,
                                                limit=limit,
                                                norm_type=norm_type)
        self.train()

    def forward(self, x, action):  # x is state
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.network(stacked_image)
        if self.residual:
            next_state = next_state + x
        next_state = F.relu(next_state)
        if self.renormalize:
            next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward


class RewardPredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_size=1,
                 pixels=36,
                 limit=300,
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 256),
                  nn.ReLU(),
                  nn.Linear(256, limit*2 + 1)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)
