Official code for *Unsupervised Salient Patch Selection for Data-Efficient Reinforcement Learning*
# Package installation:
```
# Since we made some small modification to the original rlpyt code, we put this package here for simplicity.
unzip ./rlpyt.zip
cd ./rlpyt  # the dir that has a setup.py
pip install -e .

pip install atari-py
pip install opencv-python
pip install PyYAML
pip install wandb
pip install gym
pip install scikit-learn
pip install stable-baselines3
pip install timm
pip install kornia
pip install PyPrind
pip install 'gym[atari]'
pip install git+https://github.com/mila-iqia/atari-representation-learning.git

cd <path to put atari rom>
wget http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html
unrar <name of downloaded ROMs>
python -m atari_py.import_roms <path to unrared ROMs>
```
# MAE pre-train:
Pre-train a MAE and save its checkpoint at the end of training:

```python ./Rainbow_rlpyt/main_pretrain.py --norm_pix_loss True --num_eval_data 20 --save_checkpoint True --save_last_checkpoint_only True --visual_eval True --eval_epoch 50 --model 'custom' --epoch 100 --patch_size 8 --image_size 96 --embed_dim 64 --quick_log True --kp_ext_loss_weight 10 --use_kp False --use_reverse_kp False --use_aug False --loss_all True --kp_around False --mask_ratio 0.75 --depth 3 --num_heads 4 --save_exclude_decoder False --eval_kps False --decoder_depth 3 --decoder_embed_dim 128 --decoder_num_heads 8 --device [DEVICE] --num_training_data [#PRETRAINING_DATA] --env_name [GAME_NAME]```
## For frostbite:
```--num_training_data 5000  --env_name frostbite```
## For GAME_NAME in [msPacman, seaquest, battleZone]:
```--num_training_data 50000  --env_name [GAME_NAME]```


# To reproduce DE-rainbow:
```python ./Rainbow_rlpyt/run.py --min-steps-learn 1600 --eps-steps 1601 --target-update-interval 2000 --n-step 20 --pri-beta-steps 100000 --eval_itr0 False --eval_n_envs 50 --renormalize 0 --replay-ratio 32 --dqn-hidden-size 256 --encoder_type ConvDataEfficient --game [GAME_NAME]```


# To run SPIRL:
## 100K:
```python ./Rainbow_rlpyt/run.py --min-steps-learn 1600 --eps-steps 1601 --target-update-interval 2000 --n-step 20 --pri-beta-steps 100000 --obs_type vitRec --imagesize 96 --grayscale False --eval_itr0 False --eval_n_envs 50 --renormalize 0 --replay-ratio 32 --dqn-hidden-size 256 --encoder_type AttentionModel --mae_pretrained_path ./ckpts/[#PRETRAINING_DATA]_[GAME_NAME].pth --game [GAME_NAME] --feature_type x --attention_depth 1 --attention_num_heads 8 --attention_one_frame_one_cls False --attention_global_pool False --attention_norm_out False --attention_proj_embed_dim 32 --replay_size 100000 --n-steps 100000 --min_mask_ratio [RATIO] --rec_dynamic_45_seg 2 --rec_dynamic_45_xscale 1 --rec_thres_pad True```

## 400K:
```python ./Rainbow_rlpyt/run.py --min-steps-learn 1600 --eps-steps 1601 --target-update-interval 2000 --n-step 20 --pri-beta-steps 400000 --obs_type vitRec --imagesize 96 --grayscale False --eval_itr0 False --eval_n_envs 50 --renormalize 0 --replay-ratio 32 --dqn-hidden-size 256 --encoder_type AttentionModel --mae_pretrained_path ./ckpts/[#PRETRAINING_DATA]_[GAME_NAME].pth --game [GAME_NAME] --feature_type x --attention_depth 1 --attention_num_heads 8 --attention_one_frame_one_cls False --attention_global_pool False --attention_norm_out False --attention_proj_embed_dim 32 --replay_size 100000 --n-steps 400000 --min_mask_ratio [RATIO] --rec_dynamic_45_seg 2 --rec_dynamic_45_xscale 1 --rec_thres_pad True --num-logs 8 --batch-t 4 --batch-b 1```

For different games the maximal ratio of selected patches (i.e. min_mask_ratio) are different:
### frostbite:
```--min_mask_ratio 0.65```
### msPacman
```--min_mask_ratio 0.7```
### seaquest
```--min_mask_ratio 0.8```
### battleZone
```--min_mask_ratio 0.7```

# Visualization (e.g. fig 2 & 3 in paper):
1. Collect env images for visualiztion to `./data_random`:
   ```python ./Rainbow_rlpyt/collect_random_traj.py --env_name [GAME_NAME]```
2. Visualize:
   ```python ./Rainbow_rlpyt/eval_ckpt.py --mae_pretrained_path ./ckpts/[#PRETRAINING_DATA]_[GAME_NAME].pth --compare_data_dir ./data_random --compare_data_dir [#PATH_TO_SAEV]```