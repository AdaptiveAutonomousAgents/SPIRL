"""
Modifies the default rlpyt AtariEnv to be closer to DeepMind's setup,
tries to follow Kaixin/Rainbow's env for the most part.
"""
import numpy as np
import os
import atari_py
import cv2
import torch
import pdb
from collections import namedtuple
from gym.utils import seeding
from atariari.benchmark.wrapper import ram2label

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo

from .keypoints import GET_KP_POS, NUM_RAM_KPS
from .rainbow_utils import ALE2AARI_GAME_NAME_MAP, RecThresObs
from .vit_policy import VisualAttention


EnvInfo = namedtuple("EnvInfo", ["game_score", "traj_done"])


class AtariTrajInfo(TrajInfo):
    """TrajInfo class for use with Atari Env, to store raw game score separate
    from clipped reward signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.GameScore = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.GameScore += getattr(env_info, "game_score", 0)


class AtariEnv(Env):
    """An efficient implementation of the classic Atari RL envrionment using the
    Arcade Learning Environment (ALE).

    Output `env_info` includes:
        * `game_score`: raw game score, separate from reward clipping.
        * `traj_done`: special signal which signals game-over or timeout, so that sampler doesn't reset the environment when ``done==True`` but ``traj_done==False``, which can happen when ``episodic_lives==True``.

    Always performs 2-frame max to avoid flickering (this is pretty fast).

    Screen size downsampling is done by cropping two rows and then
    downsampling by 2x using `cv2`: (210, 160) --> (80, 104).  Downsampling by
    2x is much faster than the old scheme to (84, 84), and the (80, 104) shape
    is fairly convenient for convolution filter parameters which don't cut off
    edges.

    The action space is an `IntBox` for the number of actions.  The observation
    space is an `IntBox` with ``dtype=uint8`` to save memory; conversion to float
    should happen inside the agent's model's ``forward()`` method.

    (See the file for implementation details.)


    Args:
        game (str): game name
        frame_skip (int): frames per step (>=1)
        num_img_obs (int): number of frames in observation (>=1)
        clip_reward (bool): if ``True``, clip reward to np.sign(reward)
        episodic_lives (bool): if ``True``, output ``done=True`` but ``env_info[traj_done]=False`` when a life is lost
        max_start_noops (int): upper limit for random number of noop actions after reset
        repeat_action_probability (0-1): probability for sticky actions
        horizon (int): max number of steps before timeout / ``traj_done=True``
    """

    def __init__(self,
                 game,
                 frame_skip=4,  # Frames per step (>=1).
                 num_img_obs=4,  # Number of (past) frames in observation (>=1).
                 clip_reward=True,
                 episodic_lives=True,
                 max_start_noops=30,
                 repeat_action_probability=0.,
                 horizon=27000,
                 stack_actions=0,
                 grayscale=True,
                 imagesize=84,
                 seed=42,
                 id=0,
                 obs_type='image',
                 obs_args=None,
                 dict_obs=False,
                 ):
        save__init__args(locals(), underscore=True)
        # ALE
        game_path = atari_py.get_game_path(game)
        if not os.path.exists(game_path):
            raise IOError("You asked for game {} but path {} does not "
                " exist".format(game, game_path))
        self.ale = atari_py.ALEInterface()
        self.seed(seed, id)
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.ale.loadROM(game_path)
        if self._game in ALE2AARI_GAME_NAME_MAP:
            self._game = ALE2AARI_GAME_NAME_MAP[self._game]

        # Spaces
        self.stack_actions = stack_actions
        self._action_set = self.ale.getMinimalActionSet()
        self._action_space = IntBox(low=0, high=len(self._action_set))
        if self.stack_actions:
            raise NotImplementedError  # Should check actions when reset, or _has_fire
            self.channels += 1
        
        self.image_obs = (obs_type == 'image')
        self.ram_obs = (obs_type == 'ram')
        self.ram_kp_pos_obs = (obs_type == 'ramKpPos')  # extract keypoint position, w/o velocity
        self.ram_kp_info_obs = (obs_type == 'ramKpInfo')  # extract keypoint position, w/ other information
        self.vit_rec_obs = (obs_type == 'vitRec')  # select keypoint patches based on reconstruction error
        self.vit_ram_obs = (obs_type == 'vitRam')  # select keypoint patches based on ram 

        if self.image_obs:
            self.channels = 1 if grayscale else 3
            self.grayscale = grayscale
            self.imagesize = imagesize
            obs_shape = (num_img_obs, self.channels, imagesize, imagesize)
            self._max_frame = self.ale.getScreenGrayscale() if self.grayscale \
                              else self.ale.getScreenRGB()
        elif self.ram_obs:
            obs_shape = (num_img_obs, self.ale.getRAMSize())
            self._max_frame = self.ale.getRAM()
        elif self.ram_kp_pos_obs or self.ram_kp_info_obs:
            obs_shape = (num_img_obs, NUM_RAM_KPS[self._game] * 2)  # for pong: 3 (x, y) position: 2 paddles, 1 ball
            self._max_frame = self.ale.getRAM()
        elif self.vit_rec_obs:
            self.grayscale = grayscale
            self.features_extractor = self._obs_args['features_extractor']
            features_extractor_kwargs = self._obs_args['features_extractor_kwargs']
            fake_input = torch.randn((features_extractor_kwargs['in_chans'],\
                                      features_extractor_kwargs['img_size'],\
                                      features_extractor_kwargs['img_size']))[None]
            # NOTE: fake_input doesn't require fake_input's data range to be [0, 1]
            with torch.no_grad():
                fake_output = self.features_extractor.forward_one_frame(
                    img=fake_input.to(self._obs_args['device']))
            if self.features_extractor.feat_all or self.features_extractor.feat_x:
                if (self.features_extractor.rec_thres is not None) or\
                   (self.features_extractor.rec_weight is not None) or\
                   (self.features_extractor.rec_dynamic_deltx is not None) or\
                   (self.features_extractor.rec_dynamic_k_seg is not None) or\
                   (self.features_extractor.rec_dynamic_45_seg is not None):
                    if self._obs_args['rec_thres_pad']:
                        obs_shape = (num_img_obs,
                                     self.features_extractor.max_len_keep,
                                     fake_output.shape[-1])
                    else:
                        # Then the number of selected patches is not deterministic, so we declear a maximum shape space
                        # the memory required should be 1.5 times larger than image obs
                        obs_shape = (num_img_obs,
                                     self.features_extractor.max_len_keep + 1,  # extra 1 for len_keep
                                     fake_output.shape[-1])
                    # print(f'************************self.features_extractor.max_len_keep: {self.features_extractor.max_len_keep}')
                else:
                    obs_shape = (num_img_obs, fake_output.shape[-2], fake_output.shape[-1])
            else:
                obs_shape = (num_img_obs, fake_output.shape[-1])
            self._max_frame = self.ale.getScreenGrayscale() if self.grayscale \
                              else self.ale.getScreenRGB()
        elif self.vit_ram_obs:
            if self._game.startswith('p'):
                assert self._game == 'pong'
            self.grayscale = grayscale
            self.features_extractor = self._obs_args['features_extractor']
            features_extractor_kwargs = self._obs_args['features_extractor_kwargs']

            fake_input_img = torch.randn((features_extractor_kwargs['in_chans'],\
                                            features_extractor_kwargs['img_size'],\
                                            features_extractor_kwargs['img_size']))[None]
            # fake_input_kp_pos = torch.as_tensor(sample['kp_pos'][None]).long()  # new axis added at begining (the dim for n_envs) 
            fake_input_kp_pos = torch.randint(low=0, high=self.features_extractor.num_patch_per_row, \
                                              size=(1, NUM_RAM_KPS[self._game], 2))  # dtype: th.int64
                                              # new axis added at begining (the dim for n_envs)
            with torch.no_grad():
                fake_output = self.features_extractor.forward_one_frame(
                    img=fake_input_img.to(self._obs_args['device']),
                    kp_pos=fake_input_kp_pos.to(self._obs_args['device']),
                    )
            if self.features_extractor.feat_all or self.features_extractor.feat_x:
                obs_shape = (num_img_obs, fake_output.shape[-2], fake_output.shape[-1])
            else:
                obs_shape = (num_img_obs, fake_output.shape[-1])
            self._max_frame = self.ale.getScreenGrayscale() if self.grayscale \
                              else self.ale.getScreenRGB()
            self._ram = self.ale.getRAM()
            self._frame_kp_pos = np.zeros((NUM_RAM_KPS[self._game], 2)).astype('uint8')
        else:
            raise NotImplementedError

        if self.vit_rec_obs or self.vit_ram_obs:
            # here the low & high will be used when call sample()
            self._observation_space = FloatBox(low=-1000.0, high=1000.0,  # +-1000 is just a safe value (no actual sense)
                                               shape=obs_shape, dtype="float32")
            # TODO: if want to propogate gradient in the env wrapper, should check
            #       - env part (e.g. use th.tensor instead of np.array, check gradient for self._obs)
            #       - policy part (e.g. some parts may use no_grad())
            # NOTE: if move features_extractor to models, then one frame could be sampled many times, and the calculation cost will increase
            self._obs = np.zeros(shape=obs_shape, dtype="float32")
        else:
            self._observation_space = IntBox(low=0, high=255, shape=obs_shape,
                                             dtype="uint8")  # NOTE: use uint8 instead of float, to save memory
            self._obs = np.zeros(shape=obs_shape, dtype="uint8")
        self._raw_frame_1 = self._max_frame.copy()  # deepcopy, _max_frame and _raw_framw_1 have different memory address
        self._raw_frame_2 = self._max_frame.copy()

        self.dict_obs = dict_obs

        # Settings
        self._has_fire = "FIRE" in self.get_action_meanings()
        if self._has_fire:
            # same assertation as stable-baseline3
            assert self.get_action_meanings()[1] == "FIRE"
            # assert len(env.unwrapped.get_action_meanings()) >= 3
        self._has_up = "UP" in self.get_action_meanings()
        self._horizon = int(horizon)
        self.reset()

    def seed(self, seed=None, id=0):
        _, seed1 = seeding.np_random(seed)
        if id > 0:
            seed = seed*100 + id
        self.np_random, _ = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)

    def reset(self):
        """Performs hard reset of ALE game."""
        self.ale.reset_game()
        self._reset_obs()
        self._life_reset()
        if self._max_start_noops > 0:
            for _ in range(self.np_random.randint(1, self._max_start_noops + 1)):
                self.ale.act(0)
                if self._check_life():
                    self.reset()
        if self._has_fire:
            self.ale.act(1)  # fire
            self._update_obs(action=1)
        else:
            self._update_obs(action=0)  # (don't bother to populate any frame history)
        self._step_counter = 0
        return self.get_obs()

    def step(self, action):
        a = self._action_set[action]
        game_score = np.array(0., dtype="float32")
        for _ in range(self._frame_skip - 1):
            game_score += self.ale.act(a)
        self._get_screen(1)  # put the current frame into self._raw_frame_1
        game_score += self.ale.act(a)
        lost_life = self._check_life()  # Advances from lost_life state.
        if lost_life and self._episodic_lives:
            self._reset_obs()  # Internal reset.
            if self._has_fire:
                self.ale.act(1)  # fire 
                # NOTE: don't know why in stable baseline3, follow acting action 2
        self._update_obs(action)  # put the frame after taking action a into self._raw_frame_2, and update self._obs
        reward = np.sign(game_score) if self._clip_reward else game_score
        # True game over
        game_over = self.ale.game_over() or self._step_counter >= self.horizon
        # if episodic_lives==True, then lost one life will also lead to done=True
        done = game_over or (self._episodic_lives and lost_life)
        info = EnvInfo(game_score=game_score, traj_done=game_over)
        self._step_counter += 1
        # in rlpyt.samplers.buffer.get_example_outputs, have torchify_buffer
        return EnvStep(self.get_obs(), reward, done, info)

    def render(self, wait=10, show_full_obs=False):
        """Shows game screen via cv2, with option to show all frames in observation."""
        if not self.image_obs:
            raise NotImplementedError()
        img = self.get_obs()
        if show_full_obs:
            shape = img.shape
            img = img.reshape(shape[0] * shape[1], shape[2])
        else:
            img = img[-1]
        cv2.imshow(self._game, img)
        cv2.waitKey(wait)

    def get_obs(self):
        return self._obs.copy()

    ###########################################################################
    # Helpers

    def _get_screen(self, frame=1):
        frame = self._raw_frame_1 if frame == 1 else self._raw_frame_2
        if self.image_obs or self.vit_rec_obs:
            if self.grayscale:
                self.ale.getScreenGrayscale(frame)
            else:
                self.ale.getScreenRGB(frame)
        elif self.ram_obs or self.ram_kp_pos_obs or self.ram_kp_info_obs:
            self.ale.getRAM(frame)  # TODO: check the performance the use the latest ram instead of maximum of 2 consecutive frames
        elif self.vit_ram_obs:
            if self.grayscale:
                self.ale.getScreenGrayscale(frame)
            else:
                self.ale.getScreenRGB(frame)
            self.ale.getRAM(self._ram)  # NOTE: directly use the latest ram, instead of maximum ram for 2 consecutive frames
            label_dict = ram2label(self._game, self._ram)
            ram_kp_pos = GET_KP_POS[self._game](info=label_dict,
                raw_ram=False, image_size=self._imagesize)
            ram_kp_pos = ram_kp_pos / self._obs_args['features_extractor_kwargs']['patch_size']
            ram_kp_pos = ram_kp_pos.astype(int)
            if self._game.startswith('p'):  # pong
                patch_per_row = self.features_extractor.num_patch_per_row - 1
                extra_kp_pos = []
                for dy in [1, 2]:  # mark two more keypoints below ram_kp for two player (since the ram_kp always marked at the top of two paddles)
                    extra_kp_pos.append([ram_kp_pos[0][0], min(ram_kp_pos[0][1] + dy, patch_per_row)])  # player paddle
                    extra_kp_pos.append([ram_kp_pos[1][0], min(ram_kp_pos[1][1] + dy, patch_per_row)])  # enemy paddle
                ram_kp_pos = np.concatenate((ram_kp_pos, extra_kp_pos), axis=0)

            self._frame_kp_pos = np.array(ram_kp_pos, dtype='uint8')
        else:
            raise NotImplementedError

    def _update_obs(self, action):
        """Max of last two frames; crop two rows; downsample by 2x."""
        self._get_screen(2)  # get obs and put it in self._raw_frame_2

        if self.image_obs:
            np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
            img = cv2.resize(self._max_frame, (self.imagesize, self.imagesize), cv2.INTER_LINEAR)
            if len(img.shape) == 2:
                img = img[np.newaxis]
            else:
                img = np.transpose(img, (2, 0, 1))  # original: h,w,c, now: c,h,w
            if self.stack_actions:
                action = int(255.*action/self._action_space.n)
                action = np.ones_like(img[:1])*action
                img = np.concatenate([img, action], 0)
            # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
            self._obs = np.concatenate([self._obs[1:], img[np.newaxis]])
        elif self.ram_obs:
            # TODO: for ram, here we should just use the ram of the last frame, instead of maximum of 2 frames
            #       e.g. ram = self._raw_frame_2.copy()
            np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
            ram = self._max_frame.copy()

            if self.stack_actions:
                raise NotImplementedError("for RAM input, don't consider stack_action for now")
            # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
            self._obs = np.concatenate([self._obs[1:], ram[np.newaxis]])
        elif self.ram_kp_pos_obs or self.ram_kp_info_obs:  # keypoints' position as input
            # TODO: for ram, here we should just use the ram of the last frame, instead of maximum of 2 frames
            #       e.g. ram = self._raw_frame_2.copy()
            np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
            ram = self._max_frame.copy()
            if self.stack_actions:
                raise NotImplementedError("for RAM input, don't consider stack_action for now")
            label_dict = ram2label(self._game, ram)
            kp_pos = GET_KP_POS[self._game](info=label_dict, raw_ram=self._obs_args['raw_ram_input'])
            kp_pos = np.reshape(kp_pos, (-1))
            self._obs = np.concatenate([self._obs[1:], kp_pos[np.newaxis]])
        elif self.vit_rec_obs:
            np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
            img = cv2.resize(self._max_frame, (self._imagesize, self._imagesize), cv2.INTER_LINEAR)
            if len(img.shape) == 2:  # grayscale image
                img = img[np.newaxis]
            else:
                img = np.transpose(img, (2, 0, 1))  # original: h,w,c, now: c,h,w
            img = img[np.newaxis]  # add a dimension for #batch
            with torch.no_grad():  # TODO: should consider gradient for unfrozen modules like norm layer
                feature = self.features_extractor.forward_one_frame(
                    img=torch.tensor(img, dtype=torch.float32, device=self._obs_args['device'])/255.0)
            if self.stack_actions:
                raise NotImplementedError
            # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
            if self.dict_obs:
                feat = np.zeros_like(self._obs[0:1])  # shape (1, max_len_keep+1, num_embed)
                len_keep = feature.shape[-2]
                feat[0, :len_keep] = feature.cpu()
                if not self._obs_args["rec_thres_pad"]:
                    feat[0, -1] = len_keep
                self._obs = np.concatenate([self._obs[1:], feat])
            else:
                self._obs = np.concatenate([self._obs[1:], feature.cpu()])
        elif self.vit_ram_obs:
            np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
            img = cv2.resize(self._max_frame, (self._imagesize, self._imagesize), cv2.INTER_LINEAR)
            if len(img.shape) == 2:  # grayscale image
                img = img[np.newaxis]
            else:
                img = np.transpose(img, (2, 0, 1))  # original: h,w,c, now: c,h,w
            img = img[np.newaxis]  # add a dimension for #batch
            kp_pos = self._frame_kp_pos[np.newaxis]
            with torch.no_grad():
                feature = self.features_extractor.forward_one_frame(
                    img=torch.tensor(img, dtype=torch.float32, device=self._obs_args['device'])/255.0,
                    kp_pos=torch.tensor(kp_pos, dtype=torch.int64, device=self._obs_args['device']))
            if self.stack_actions:
                raise NotImplementedError
            # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
            self._obs = np.concatenate([self._obs[1:], feature.cpu()])
        else:
            raise NotImplementedError

    def _reset_obs(self):
        self._obs[:] = 0
        if self.dict_obs and not self._obs_args["rec_thres_pad"]:
            self._obs[:, -1, :] = 1
        self._max_frame[:] = 0
        self._raw_frame_1[:] = 0
        self._raw_frame_2[:] = 0

    def _check_life(self):
        lives = self.ale.lives()
        lost_life = (lives < self._lives) and (lives > 0)
        if lost_life:
            self._life_reset()
        return lost_life

    def _life_reset(self):
        self.ale.act(0)  # (advance from lost life state)
        self._lives = self.ale.lives()

    ###########################################################################
    # Properties

    @property
    def game(self):
        return self._game

    @property
    def frame_skip(self):
        return self._frame_skip

    @property
    def num_img_obs(self):
        return self._num_img_obs

    @property
    def clip_reward(self):
        return self._clip_reward

    @property
    def max_start_noops(self):
        return self._max_start_noops

    @property
    def episodic_lives(self):
        return self._episodic_lives

    @property
    def repeat_action_probability(self):
        return self._repeat_action_probability

    @property
    def horizon(self):
        return self._horizon

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
# RIGHTFIRE and LEFTFIRE do RIGHT + FIRE and LEFT + FIRE

ACTION_INDEX = {v: k for k, v in ACTION_MEANING.items()}
