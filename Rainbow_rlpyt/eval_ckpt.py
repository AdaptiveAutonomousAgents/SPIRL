
"""
Parallel sampler version of Atari DQN.  Increasing the number of parallel
environmnets (sampler batch_B) should improve the efficiency of the forward
pass for action sampling on the GPU.  Using a larger batch size in the algorithm
should improve the efficiency of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
# from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.logging import logger

import wandb
import torch
import numpy as np
import pdb
import os
import yaml
import time

from spirl.models import SPRCatDqnModel
from spirl.rlpyt_utils import OneToOneSerialEvalCollector, SerialSampler, MinibatchRlEvalWandb, delete_ind_from_array
from spirl.algos import SPRCategoricalDQN
from spirl.agent import SPRAgent
from spirl.rlpyt_atari_env_eval import AtariEnvEval
from spirl.rainbow_utils import set_config, ARG2ALE_GAME_NAME_MAP, str2bool

def build_and_train(game="seaquest", run_ID=0, args=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = AtariEnvEval
    config = set_config(args, game)
    config["eval_env"]["disturb"] = args.eval_env_disturb
    config["eval_env"]["num_pixel"] = args.num_pixel
    config["eval_env"]["num_black_patch"] = args.num_black_patch
    config["eval_env"]["num_pixel_ratio"] = args.num_pixel_ratio
    config["eval_env"]["vit_loss_type"] = args.vit_loss_type

    sampler = SerialSampler(  # TODO: maybe can try to use parallel cpu samplers? see example_3.py
        EnvCls=env,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=config["env"],
        eval_env_kwargs=config["eval_env"],
        batch_T=config['sampler']['batch_T'],
        batch_B=config['sampler']['batch_B'],
        max_decorrelation_steps=0,
        eval_CollectorCls=OneToOneSerialEvalCollector,
        eval_n_envs=config["sampler"]["eval_n_envs"],
        eval_max_steps=config['sampler']['eval_max_steps'],
        eval_max_trajectories=config["sampler"]["eval_max_trajectories"],
    )
    args.discount = config["algo"]["discount"]
    algo = SPRCategoricalDQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
    agent = SPRAgent(ModelCls=SPRCatDqnModel,
                     model_kwargs=config["model"], **config["agent"])

    if args.cuda_idx < 0:
        cuda_idx = None
    else:
        cuda_idx = args.cuda_idx

    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args.n_steps,
        affinity=dict(cuda_idx=cuda_idx),
        log_interval_steps=args.n_steps//args.num_logs,
        seed=args.seed,
        final_eval_only=args.final_eval_only,
        eval_itr0=args.eval_itr0,
    )
    runner.startup()

    eval_time = -time.time()
    eval_collector = runner.sampler.eval_collector
    traj_infos = [eval_collector.TrajInfoCls() for _ in range(len(eval_collector.envs))]
    completed_traj_infos = list()
    observations = list()
    for env in eval_collector.envs:
        observations.append(env.reset())
    observation = buffer_from_example(observations[0], len(eval_collector.envs))
    for b, o in enumerate(observations):
        observation[b] = o
    action = buffer_from_example(eval_collector.envs[0].action_space.null_value(),
                                    len(eval_collector.envs))
    reward = np.zeros(len(eval_collector.envs), dtype="float32")
    obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))

    if (args.agent_ckpt_path is not None) and (args.agent_ckpt_path != "None"):
        if cuda_idx is None:
            ckpt = torch.load(args.agent_ckpt_path, map_location=torch.device('cpu'))
        else:
            ckpt = torch.load(args.agent_ckpt_path)

    msg = eval_collector.agent.search.network.load_state_dict(ckpt, strict=False)
    print(f"{'*'*10}checkpoint load information for pretrained agent model{'*'*10}")
    print(msg)
    eval_collector.agent.reset()
    eval_collector.agent.eval_mode(1)  # epsilon = eps_eval if itr > 0 else 1
    live_envs = list(range(len(eval_collector.envs)))
    for t in range(eval_collector.max_T):
        act_pyt, agent_info = eval_collector.agent.step(obs_pyt, act_pyt, rew_pyt)
        action = numpify_buffer(act_pyt)

        b = 0
        while b < len(live_envs):  # don't want to do a for loop since live envs changes over time
            env_id = live_envs[b]
            o, r, d, env_info = eval_collector.envs[env_id].step(action[b])
            traj_infos[env_id].step(observation[b],
                                    action[b], r, d,
                                    agent_info[b], env_info)
            if getattr(env_info, "traj_done", d):
                completed_traj_infos.append(traj_infos[env_id].terminate(o))

                observation = delete_ind_from_array(observation, b)
                reward = delete_ind_from_array(reward, b)
                action = delete_ind_from_array(action, b)
                obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))

                del live_envs[b]
                b -= 1  # live_envs[b] is now the next env, so go back one.
            else:
                observation[b] = o
                reward[b] = r

            b += 1

            if (eval_collector.max_trajectories is not None and
                    len(completed_traj_infos) >= eval_collector.max_trajectories):
                logger.log("Evaluation reached max num trajectories "
                            f"({eval_collector.max_trajectories}).")
                break
                # return completed_traj_infos
        if (eval_collector.max_trajectories is not None and
                len(completed_traj_infos) >= eval_collector.max_trajectories):
            logger.log("Evaluation reached max num trajectories "
                        f"({eval_collector.max_trajectories}).")
            break

    if t == eval_collector.max_T - 1:
        logger.log("Evaluation reached max num time steps "
                    f"({eval_collector.max_T}).")
    eval_time += time.time()
    eval_traj_infos = completed_traj_infos
    runner.log_diagnostics(0, eval_traj_infos, eval_time)
    quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='seaquest',
                        choices=['msPacman', 'frostbite', 'seaquest', 'battleZone'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grayscale', type=str2bool, default=True)
    parser.add_argument('--obs_type', type=str, default='image', choices=['image', 'ram', 'ramKpPos', 'ramKpInfo', \
                        'vitRec', 'vitRam'])
    parser.add_argument('--raw_ram_input', type=str2bool, default=False,
                        help="(If use --obs_type=ramKpPos) if False, will force the position to [0, H or W]")

    parser.add_argument("--mae_pretrained_path", type=str,
                        help="path to load pretrained mae; if None, won't load ckpt for ViT feature extractor")
    parser.add_argument("--agent_ckpt_path", type=str,
                        help="path to load pretrained attention RL.")

    parser.add_argument("--eval_env_disturb", type=str, default=None,
                        choices=["singlePixel0", "singlePixelResized","fixWhite0", "fixWhiteResized",
                         "fixBlackPatch", "randBlackPatch"],
                        help="type for noise added to eval_env.")
    parser.add_argument("--num_pixel", type=int, default=None,
                        help="ratio of number of single pixels")
    parser.add_argument("--num_black_patch", type=int, default=0)
    parser.add_argument("--num_pixel_ratio", type=int, default=0,
                        help="ratio of number of single pixels")
    parser.add_argument('--vit_loss_type', default='L2', type=str, choices=['L2', 'L1', 'L2trim1',
                            "L2trim2", "L2trim3", "L2trim4", "L2trim5", "L2trim6", "L2trim7",
                            "L2trim8", "L2trim9", "L2trim10", "L2trim12","L2trim14","L2trim16",
                            "L2trim18","L2trim20","L2trim22","L2trim24","L2trim26","L2trim28",
                            "L2trim30"])
    parser.add_argument("--feature_type", type=str, help='Feature type for vit embedding',
                        default='normAvg', choices=['cls', 'all', 'x', 'xCls', 'concat',
                                                    'normAvg', 'avg', 'normAvgCls', 'avgCls',
                                                    'normMax', 'max', 'normMaxCls', 'maxCls',
                                                    'normSum', 'sum', 'normSumCls', 'sumCls',
                                                    'avgMaxCls', 'avgMax', 'sumMaxCls', 'sumMax'])
    parser.add_argument("--mask_ratio", type=float, default=0.9, help='For MAE reconstruction error map')
    parser.add_argument("--mask_ratio_random_bound", type=float, default=None, help='mask ratio for selecting important patches, \
                        then select more random patches up to mask_ratio_random_bound')
    parser.add_argument("--rec_thres", type=float, default=None, help='if rec_thres in (0, 1], for vitRec, select \
                        key patches based on error threshold')
    parser.add_argument("--rec_weight", type=float, default=None, help='if rec_weight in (0, 1], for vitRec, select \
                        key patches based on error weight')
    parser.add_argument("--rec_dynamic_deltx", type=float, default=None, help='if rec_dynamic_deltx is not None, for vitRec, select \
                        key patches based on dynamic threshold')
    parser.add_argument("--rec_dynamic_ma", type=int, default=None,  # original default=7
                        help='if rec_dynamic_deltx is not None, for vitRec, select \
                        key patches based on dynamic threshold with this moving average window size')
    parser.add_argument("--rec_dynamic_k_seg", type=int, default=None,  # original default=15
                        help='if rec_dynamic_k_seg is not None, for vitRec, select \
                        key patches based on dynamic threshold (slope for segments)')
    parser.add_argument("--rec_dynamic_45_seg", type=int, default=None,  # default should be 2 (= consecutive points)
                        help='if rec_dynamic_45_seg is not None, for vitRec, select \
                        key patches based on dynamic threshold (45 degree, segments)')
    parser.add_argument("--rec_dynamic_45_xscale", type=int, default=None,  # default should be 1
                        help='if rec_dynamic_45_seg is not None, delt x =xscale/#patches')

    parser.add_argument("--rec_thres_div", type=str2bool, default=False, help='if not rec_thres_pad, rec_thres_div will scale the \
                        global embedding for each graph by the number of nodes in that graph')
    parser.add_argument('--rec_thres_pad', type=str2bool, default=False, help="If true, padding extra obs embeddings as zero \
                        (to be able to train RL encoder via patch)")  # TODO: change name to rec_pad
    parser.add_argument("--min_mask_ratio", type=float, default=None,  # original default=0.6
                        help="To restrict the maximum number of kept patches when rec_thres is not None")
    parser.add_argument("--rec_idx_bc", type=int, default=128, help='batch size to calculate reconstruction error')
    parser.add_argument("--add_pos_type", type=str, default='addSS',
                        choices=['addSS', 'catXY', 'catXYMap', 'addSScatXY', 'addSScatXYMap', 'none'],
                        help="type of position embeddings added after attention blocks")
    parser.add_argument('--check_kp', type=str2bool, default=False, help="If true, save frames marked with selected kps")
    parser.add_argument('--use_tf_writer', type=str2bool, default=False, help="If true, save log to tf summary writer")
    parser.add_argument('--eval_n_envs', type=int, default=100)
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--imagesize', type=int, default=84)
    parser.add_argument('--n-steps', type=int, default=100000)
    parser.add_argument('--dqn-hidden-size', type=int, default=256)
    parser.add_argument('--target-update-interval', type=int, default=1)
    parser.add_argument('--target-update-tau', type=float, default=1.)
    parser.add_argument('--momentum-tau', type=float, default=0.01)
    parser.add_argument('--batch-b', type=int, default=1, help="(For sampler) batch size, Number of separate trajectory segments (i.e. # env instances)")
    parser.add_argument('--batch-t', type=int, default=1, help="(For sampler) Number of time steps, >=1")
    parser.add_argument('--beluga', action="store_true")
    parser.add_argument('--num-logs', type=int, default=10)
    parser.add_argument('--save_ckpt', type=str2bool, default=False, help="True if save RL ckpt")
    parser.add_argument('--renormalize', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--encoder_type', type=str, default="ConvCanonical", choices=["ConvCanonical", "ConvDataEfficient", \
                        "MLPW0D1", "MLPW0D2", "MLPW0D3", "MLPW0D4", \
                        "MLPW1D1", "MLPW2D1", "MLPW1D2", "MLPW2D2", "MLPW3D2", "MLPW1D3", "MLPW2D3", \
                        "AttentionModel"])
    parser.add_argument('--attention_depth', type=int, default=1)
    parser.add_argument('--attention_recurrent', type=int, default=1)
    parser.add_argument('--attention_num_heads', type=int, default=8)
    parser.add_argument('--attention_residual', type=str2bool, default=True)
    parser.add_argument('--attention_one_frame_one_cls', type=str2bool, default=False)
    parser.add_argument('--attention_proj_embed_dim', type=int, default=-1, help="if >0, will project it to another dimension")
    parser.add_argument('--attention_one_frame_one_proj', type=str2bool, default=False)
    parser.add_argument('--attention_global_pool', type=str2bool, default=False)
    parser.add_argument('--attention_norm_out', type=str2bool, default=True)
    
    parser.add_argument('--replay-ratio', type=int, default=64, help="should be divisible by batch_size")
    parser.add_argument('--replay_size', type=int, default=1000000, help="replay buffer size")
    parser.add_argument('--unbounded_replay', type=str2bool, default=True, help="use unbounded replay buffer")
    parser.add_argument('--residual-tm', type=int, default=0., help="For transition model")
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--tag', type=str, default='', help='Tag for wandb run.')
    parser.add_argument('--norm-type', type=str, default='bn', choices=["bn", "ln", "in", "none"], help='Normalization for the transition model')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability in convnet.')
    parser.add_argument('--distributional', type=int, default=1)
    parser.add_argument('--delta-clip', type=float, default=1., help="Huber Delta")
    parser.add_argument('--prioritized-replay', type=int, default=1)
    parser.add_argument('--momentum-encoder', type=int, default=1)
    parser.add_argument('--shared-encoder', type=int, default=0)
    parser.add_argument('--noisy-nets', type=int, default=1)
    parser.add_argument('--noisy-nets-std', type=float, default=0.5)
    parser.add_argument('--classifier', type=str, default='q_l1', choices=["mlp", "bilinear", "q_l1", "q_l2", "none"], help='Style of NCE classifier')
    parser.add_argument('--final-classifier', type=str, default='linear', choices=["mlp", "linear", "none"], help='Style of NCE classifier')
    parser.add_argument('--eps-steps', type=int, default=2001, help="epislon-greedy step. If use noisy-nets, then shouldn't use epislon-greedy.")
    parser.add_argument('--min-steps-learn', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--eps_init', type=float, default=1.)
    parser.add_argument('--eps_final', type=float, default=0., help="If use eps-greedy, DQN use 0.1 as the final eps")
    parser.add_argument('--eps_eval', type=float, default=0.001, help="epslion when eval agent. In rlpyt code, default 0.001")
    parser.add_argument('--final-eval-only', type=int, default=1)
    parser.add_argument('--eval_itr0', type=str2bool, default=False)
    parser.add_argument('--cuda_idx', type=int, default=0, help='gpu to use; if is -1, will not use gpu')
    parser.add_argument('--max-grad-norm', type=float, default=10., help='Max Grad Norm')
    parser.add_argument('--pri-beta-steps', type=int, default=10000, help='timesteps for linear increasing for priority_replay_beta')
    args = parser.parse_args()
    
    if args.game in ARG2ALE_GAME_NAME_MAP:
        args.game = ARG2ALE_GAME_NAME_MAP[args.game]
    if not torch.cuda.is_available():
        args.cuda_idx = -1

    if args.unbounded_replay:
        args.replay_size = args.n_steps + 100000

    assert not ((args.rec_thres is not None) and (args.rec_weight is not None))

    build_and_train(game=args.game, args=args)

