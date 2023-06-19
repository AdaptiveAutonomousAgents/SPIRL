"""
To compare keypoint selection visualization with transporter and permakey,
we need to collect frames from a random policy use a same env wrapper as us.
"""
import argparse
import numpy as np
import os
import pdb
import random

from spirl.rlpyt_atari_env import AtariEnv
from spirl.rainbow_utils import str2bool, ARG2ALE_GAME_NAME_MAP

def get_args_parser():
    parser = argparse.ArgumentParser('Collect Atari frames from a random policy', add_help=False)
    parser.add_argument('--total_data', default=1000, type=int, help='total number of images to collect')
    parser.add_argument('--root_dir', default='./data_random', type=str)
    parser.add_argument('--env_name', choices=[
                'msPacman', 'frostbite', 'seaquest', 'battleZone'], type=str,
                default='seaquest')
    parser.add_argument('--seed', default=567, type=int)
    return parser


def main(args):
    env = AtariEnv(game=args.env_name, num_img_obs=1,  # num_img_obs = frame stack
                    grayscale=False, imagesize=96, seed=args.seed,
                    obs_type='image')
    num_data = 0
    env.reset()
    img_ls = []
    obs, _, done, info = env.step(0)

    while num_data < args.total_data:
        print(f'have colloect {num_data} frames...')
        env.reset()

        done = False
        while not done:
            obs, _, done, info = env.step(random.randint(0, env.action_space.n - 1))

            img_ls.append(obs[0])  # Since num_img_obs==1, no frame stack, CHW
            num_data += 1
            if num_data >= args.total_data:
                break

    # TODO: for permakey, need to /255 and transform to HWC
    dataset = np.stack(img_ls)  # [0,255], (B, C, H, W)
    num_training = int(args.total_data * 0.85)
    os.makedirs(args.root_dir, exist_ok=True)

    train_path = os.path.join(args.root_dir,'train')
    os.makedirs(train_path, exist_ok=True)
    batch_size = 2500

    num_batch = 0
    for i in range(0, num_training, batch_size):
        np.savez(os.path.join(train_path, str(num_batch)),
                 frames=dataset[i:i+batch_size])
        num_batch += 1

    test_path = os.path.join(args.root_dir,'test')
    os.makedirs(test_path, exist_ok=True)
    num_batch = 0
    for i in range(num_training, args.total_data, batch_size):
        np.savez(os.path.join(test_path, str(num_batch)),
                 frames=dataset[i:min(i+batch_size, args.total_data)])
        num_batch += 1


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.env_name in ARG2ALE_GAME_NAME_MAP:
        args.env_name = ARG2ALE_GAME_NAME_MAP[args.env_name]
    main(args)
