import pdb
import torch
from .vit_policy import VisualAttention
from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.utils.collections import namedarraytuple

# Since lhpo can not handle underscore in hyperparameters, we need this map
# GAME_NAME_MAP = {
ARG2ALE_GAME_NAME_MAP = {
    'msPacman': 'ms_pacman',
    'battleZone': 'battle_zone',
    'spaceInvaders': 'space_invaders',
}

ENV_NAME_TO_GYM_NAME = {
    "seaquest": "SeaquestNoFrameskip-v4",
    "frostbite": "FrostbiteNoFrameskip-v4",
    "ms_pacman": "MsPacmanNoFrameskip-v4",
    "battle_zone": "BattleZoneNoFrameskip-v4",
    }

ALE2AARI_GAME_NAME_MAP = {
    'ms_pacman': 'mspacman',
    'battle_zone': 'battlezone',
}

RecThresObs = namedarraytuple("RecThresObs", ["feat", "len_keep"])

def str2bool(str):
    if str.lower() == 'true':
        return True 
    elif str.lower() == 'false':
        return False
    else:
        raise NotImplementedError()

def count_parameters(model):
    print(model)
    total_params = 0
    total_params_grad = 0
    for n, p in model.named_parameters():
        print(f'{n}, requires_grad:{p.requires_grad}, shape:{p.shape}, numel:{p.numel()}')
        total_params += p.numel()
        total_params_grad += p.numel() if p.requires_grad else 0

    print(f'total_params: {total_params}, total_params_grad: {total_params_grad}')
    return total_params_grad


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def set_config(args, game):
    config = configs['ernbw']
    config['env']['game'] = config["eval_env"]["game"] = game
    config["env"]["grayscale"] = config["eval_env"]["grayscale"] = args.grayscale
    config["env"]["obs_type"] = config["eval_env"]["obs_type"] = args.obs_type
    if args.obs_type in ['ramKpPos', 'ramKpInfo']:
        obs_args = {
            'raw_ram_input': args.raw_ram_input,
        }
        model_dict_input = False
    elif args.obs_type in ['vitRec', 'vitRam']:
        if args.obs_type=='vitRec':
            kp_type = 'rec'
        elif args.obs_type=='vitRam':
            kp_type = 'ram'
        feature_extractor_ckpt = torch.load(args.mae_pretrained_path, map_location='cpu')  # NOTE: the map_location choice may cost a lot of time
        features_extractor_kwargs = {}
        for key in ['patch_size', 'embed_dim', 'depth', 'num_heads', 'in_chans',
                    'decoder_embed_dim', 'decoder_depth', 'decoder_num_heads',
                    'norm_pix_loss']:
            features_extractor_kwargs[key] = eval(f"feature_extractor_ckpt['args'].{key}") # use pretrained model's setting
        features_extractor_kwargs['kp'] = kp_type
        features_extractor_kwargs['img_size'] = args.imagesize
        features_extractor_kwargs['feature'] = args.feature_type
        features_extractor_kwargs['mask_ratio'] = args.mask_ratio
        features_extractor_kwargs['mask_ratio_random_bound'] = args.mask_ratio_random_bound
        features_extractor_kwargs['rec_thres'] = args.rec_thres
        features_extractor_kwargs['rec_weight'] = args.rec_weight
        features_extractor_kwargs['rec_dynamic_deltx'] = args.rec_dynamic_deltx
        features_extractor_kwargs['rec_dynamic_ma'] = args.rec_dynamic_ma
        features_extractor_kwargs['rec_dynamic_k_seg'] = args.rec_dynamic_k_seg
        features_extractor_kwargs['rec_dynamic_45_seg'] = args.rec_dynamic_45_seg
        features_extractor_kwargs['rec_dynamic_45_xscale'] = args.rec_dynamic_45_xscale
        
        if (args.rec_thres is not None) or (args.rec_weight is not None) \
            or (args.rec_dynamic_deltx is not None) \
            or (args.rec_dynamic_k_seg is not None) \
            or (args.rec_dynamic_45_seg is not None):
            assert args.feature_type == 'x'  # for 'all' and 'xCls', need to provide larger buffer in env
        features_extractor_kwargs['rec_idx_bc'] = args.rec_idx_bc
        features_extractor_kwargs['add_pos_type'] = args.add_pos_type
        features_extractor_kwargs['check_kp_path'] = f"./logs/vit_kp_{kp_type}" if args.check_kp else None
        features_extractor_kwargs['min_mask_ratio'] = args.min_mask_ratio
        device = 'cuda' if args.cuda_idx >= 0 else 'cpu'
        print(f'****device is {device}')
        assert feature_extractor_ckpt['args'].env_name.lower().replace("_", "") \
                == game.lower().replace("_", "")

        features_extractor = VisualAttention(**features_extractor_kwargs)
        features_extractor.load_trained_mae(feature_extractor_ckpt['model'])
        features_extractor.to(device)
        obs_args = {
            'features_extractor': features_extractor,
            'features_extractor_kwargs': features_extractor_kwargs,
            'device': device,
            'rec_thres_pad': args.rec_thres_pad,
        }
        if args.obs_type == 'vitRec' and \
            (features_extractor.feat_all or features_extractor.feat_x) and \
            ((features_extractor.rec_thres is not None) or (features_extractor.rec_weight is not None) \
                or (features_extractor.rec_dynamic_deltx is not None) \
                or (features_extractor.rec_dynamic_k_seg is not None) \
                or (features_extractor.rec_dynamic_45_seg is not None)):
            assert features_extractor.max_len_keep is not None
            model_dict_input = True
        else:
            model_dict_input = False
    else:
        model_dict_input = False
        obs_args = None
    config["env"]["obs_args"] = config["eval_env"]["obs_args"] = obs_args
    config["env"]["num_img_obs"] = config["eval_env"]["num_img_obs"] = args.framestack
    # config["eval_env"]["game"] = config["env"]["game"]
    # config["eval_env"]["grayscale"] = args.grayscale
    # config["eval_env"]["obs_type"] = args.obs_type
    # config["eval_env"]["num_img_obs"] = args.framestack
    config['env']['imagesize'] = config['eval_env']['imagesize'] = args.imagesize
    # config['eval_env']['imagesize'] = args.imagesize
    config['env']['seed'] = config['eval_env']['seed'] = args.seed
    config["env"]["dict_obs"] = config["eval_env"]["dict_obs"] = model_dict_input
    # config['eval_env']['seed'] = args.seed
    config["model"]["dueling"] = bool(args.dueling)
    config["model"]["encoder_type"] = args.encoder_type
    config["algo"]["min_steps_learn"] = args.min_steps_learn
    config["algo"]["n_step_return"] = args.n_step
    config["algo"]["batch_size"] = args.batch_size
    config["algo"]["learning_rate"] = args.lr
    config['algo']['replay_ratio'] = args.replay_ratio
    config['algo']['target_update_interval'] = args.target_update_interval
    config['algo']['target_update_tau'] = args.target_update_tau
    config['algo']['eps_steps'] = args.eps_steps
    config["algo"]["clip_grad_norm"] = args.max_grad_norm
    config['algo']['pri_alpha'] = 0.5
    config['algo']['pri_beta_steps'] = args.pri_beta_steps
    config['algo']['replay_size'] = args.replay_size
    config['optim']['eps'] = 0.00015  # default setting in Rainbow (all variants)
    config["sampler"]["eval_max_trajectories"] = args.eval_n_envs
    config["sampler"]["eval_n_envs"] = args.eval_n_envs
    config["sampler"]["eval_max_steps"] = config["sampler"]["eval_max_trajectories"]*28000  # 28k is just a safe ceiling
    config['sampler']['batch_B'] = args.batch_b
    config['sampler']['batch_T'] = args.batch_t

    config['agent']['eps_init'] = args.eps_init
    config['agent']['eps_final'] = args.eps_final
    config["model"]["noisy_nets_std"] = args.noisy_nets_std

    # if args.noisy_nets:
        # config['agent']['eps_eval'] = 0.001
    config['agent']['eps_eval'] = args.eps_eval

    # New Arguments
    config["model"]["imagesize"] = args.imagesize
    config["model"]["noisy_nets"] = args.noisy_nets
    config["model"]["distributional"] = args.distributional
    config["model"]["renormalize"] = args.renormalize
    config["model"]["dropout"] = args.dropout
    config["model"]["classifier"] = args.classifier
    config["model"]["dqn_hidden_size"] = args.dqn_hidden_size
    config["model"]["obs_type"] = args.obs_type
    config["model"]["game"] = args.game if args.game not in ALE2AARI_GAME_NAME_MAP else ALE2AARI_GAME_NAME_MAP[args.game]
    config["model"]["obs_args"] = obs_args
    config["model"]["dict_input"] = model_dict_input
    attention_args = {
        "depth": args.attention_depth,
        "recurrent": args.attention_recurrent,
        "num_heads": args.attention_num_heads,
        "one_frame_one_cls": args.attention_one_frame_one_cls,
        "proj_embed_dim": args.attention_proj_embed_dim,
        "one_frame_one_proj": args.attention_one_frame_one_proj,
        "global_pool": args.attention_global_pool,
        "norm_out": args.attention_norm_out,
        'rec_thres_pad': args.rec_thres_pad,
        'input_xCls': args.feature_type=='xCls',
        "rec_thres_div": args.rec_thres_div,
        "residual": args.attention_residual,
    }
    if args.rec_thres_div:
        assert not args.rec_thres_pad
    config["model"]["attention_args"] = attention_args
    config["algo"]["distributional"] = args.distributional
    config["algo"]["delta_clip"] = args.delta_clip
    config["algo"]["prioritized_replay"] = args.prioritized_replay

    return config