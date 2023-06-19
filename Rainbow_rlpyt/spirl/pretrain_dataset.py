import cv2
import numpy as np
import random
import pdb
import os
import matplotlib.pyplot as plt
from PIL import Image
from zipfile import ZipFile, BadZipFile
from torch.utils.data import Dataset

from .rlpyt_atari_env import AtariEnv
from .rainbow_utils import ENV_NAME_TO_GYM_NAME


class AtariKeypointDataset(Dataset):
    pass

class AtariNormalDataset(Dataset):
    def __init__(self, env_name, total_data, img_size, in_chans, seed, transpose, transform=None, save_obs=None):
        self.env_name = env_name  # e.g. pong, breakout
        self.total_data = total_data
        self.img_size = img_size
        self.in_chans = in_chans
        self.transform = transform
        self.seed = seed
        self.transpose = transpose
        self.img_ls = self.get_frames_without_kps(self.env_name, self.total_data, save_obs)
        
        assert len(self.img_ls) == self.total_data

    def __len__(self):
        return self.total_data
    
    def __getitem__(self, idx):
        return self.img_ls[idx]

    def get_frames_without_kps(self, env_name, total_data, save_obs=None):
        '''

        collect atari consecutive Atari video frames from a random policy

        total_data: total number of collected data
        '''
        img_ls = []  # original RGB images
        num_data = 0

        env = AtariEnv(game=env_name, num_img_obs=1,  # num_img_obs = frame stack
                       grayscale=(self.in_chans==1), imagesize=self.img_size, seed=self.seed,
                       obs_type='image')
        # env = make_atari_env(env_name=env_name, mode='train', colour_input=self.in_chans==3,
        #                      img_size=self.img_size, seed=self.seed, scale=True,
        #                      transpose=self.transpose)
        env.reset()
        obs, _, done, info = env.step(0)
        if self.transform:
            assert obs.dtype == 'uint8'  # so that in transforms.ToTensor() obs will be scaled from [0, 255] to [0.0, 1.0]

        if save_obs is not None:
            fig_cnt = 0

        while num_data < total_data:
            print(f'have colloect {num_data} frames...')
            env.reset()

            done = False
            while not done:
                obs, _, done, info = env.step(random.randint(0, env.action_space.n - 1))
                
                if save_obs is not None:
                    plt.imshow(obs)
                    plt.savefig(os.path.join(save_obs, f'{fig_cnt}.png'))
                    fig_cnt += 1

                if self.transform:
                    obs = self.transform(Image.fromarray(obs))

                img_ls.append(obs[0])  # Since num_img_obs==1, no frame stack, CHW
                num_data += 1
                if num_data >= total_data:
                    break
        return np.stack(img_ls).astype('float32')/255.0


class AtariOfflineDataset(Dataset):
    """
    Same setting with permakey: use rollouts of various pre-trained 
    agents in the Atari Model Zoo
    """
    def __init__(self, env_name, img_size, in_chans, split, total_data=None):
        self.env_name = env_name  # e.g. pong, breakout
        self.img_size = img_size
        self.in_chans = in_chans
        self.total_data = total_data
        self.img_ls = self.get_frames(split=split)
        # assert len(self.img_ls) == self.total_data
        
    def __len__(self):
        return len(self.img_ls)
    
    def __getitem__(self, idx):
        return self.img_ls[idx]

    def get_frames(self, split):
        env_name = ENV_NAME_TO_GYM_NAME[self.env_name]

        data_path = './projects/permakey/data/atari'
        if split == "train":
            data_dir = os.path.join(data_path, "train", env_name)
        elif split == "test":
            data_dir = os.path.join(data_path, "test", env_name)
        else:
            raise ValueError("Unknown dataset split: %s" % split)

        assert os.path.isdir(data_dir), "%s does not exist" % data_dir

        # Load files
        filenames_list = []
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:  # opens each .npz file
                filenames_list.append(os.path.join(subdir, file))

        obs_ls = []
        # creating training dataset class
        for file in filenames_list:
            try:
                with ZipFile(file, 'r') as zf:
                    data = np.load(file)
                    if self.in_chans==1:  # gray-scale
                        obs = data['observations']
                        for i in range(obs.shape[0]):
                            resized_img = cv2.resize(obs[i, :, :, 3][:, :, None],
                                                        (self.img_size, self.img_size),
                                                        interpolation=cv2.INTER_AREA)
                            transposed_img = np.einsum('hwc->chw', resized_img)
                            obs_ls.append(transposed_img.astype('float32'))

                    elif self.in_chans==3:  # atari full-sized colored frames (160, 210, 3)
                        obs = data['frames'] / 255.0
                        for i in range(obs.shape[0]):
                            resized_img = cv2.resize(obs[i],
                                                        (self.img_size, self.img_size),
                                                        interpolation=cv2.INTER_AREA)
                            transposed_img = np.einsum('hwc->chw', resized_img)
                            obs_ls.append(transposed_img.astype('float32'))
            except BadZipFile:
                print("Corrupted zip file ignored..")

        print(f'--------- len(data_loaded) from pretrained agent in Atari zoo: {len(obs_ls)} ----------')
        if self.total_data is not None:
            intv = len(obs_ls)//self.total_data
            obs_ls = obs_ls[::intv]
        return obs_ls  # [0,1] float32, CHW
