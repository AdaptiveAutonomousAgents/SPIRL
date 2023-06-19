import numpy as np
import random
import pdb
from functools import partial

# TODO: should remove this part at the end of the project
# default rendered image size for atari game
HEIGHT = 210
WEIGHT = 160


def get_keypoints_breakout(info, drift_x, drift_y, raw_ram, height=HEIGHT, weight=WEIGHT):
    if raw_ram:
        player_x = info["player_x"]
        player_y = 190 + drift_y
        ball_x = info["ball_x"]
        ball_y =info["ball_y"]
    else:
        # -1 to avoid point position in edge, which will cause error for patch position calculation
        weight = weight - 1
        height = height - 1
        player_x = min(max(info["player_x"]-drift_x, 0), weight)
        player_y = 190
        ball_x = min(max(info["ball_x"]-drift_x-10, 0), weight)
        ball_y = min(max(info["ball_y"]-drift_y, 0), height)
    kp_ls = [[player_x, player_y], [ball_x, ball_y]]
    kp_ls = np.array(kp_ls)
    return kp_ls.astype('uint8')


def get_keypoints_mspacman(info, drift_x, drift_y, raw_ram, height=HEIGHT, weight=WEIGHT, image_size=None):
    if raw_ram:
        player_x = info["player_x"]
        player_y = info["player_y"]
        fruit_x = info["fruit_x"]
        fruit_y = info["fruit_y"]
        enemy_sue_x = info["enemy_sue_x"]
        enemy_sue_y = info["enemy_sue_y"]
        enemy_inky_x = info["enemy_inky_x"]
        enemy_inky_y = info["enemy_inky_y"]
        enemy_pinky_x = info["enemy_pinky_x"]
        enemy_pinky_y = info["enemy_pinky_y"]
        enemy_blinky_x = info["enemy_blinky_x"]
        enemy_blinky_y = info["enemy_blinky_y"]
    else:
        # -1 to avoid point position in edge, which will cause error for patch position calculation
        weight = weight - 1
        height = height - 1
        player_x = min(max(info["player_x"]-drift_x, 0), weight)
        player_y = min(max(info["player_y"]-drift_y, 0), height)
        fruit_x = min(max(info["fruit_x"]-drift_x, 0), weight)
        fruit_y = min(max(info["fruit_y"]-drift_y, 0), height)
        enemy_sue_x = min(max(info["enemy_sue_x"]-drift_x, 0), weight)
        enemy_sue_y = min(max(info["enemy_sue_y"]-drift_y, 0), height)
        enemy_inky_x = min(max(info["enemy_inky_x"]-drift_x, 0), weight)
        enemy_inky_y = min(max(info["enemy_inky_y"]-drift_y, 0), height)
        enemy_pinky_x = min(max(info["enemy_pinky_x"]-drift_x, 0), weight)
        enemy_pinky_y = min(max(info["enemy_pinky_y"]-drift_y, 0), height)
        enemy_blinky_x = min(max(info["enemy_blinky_x"]-drift_x, 0), weight)
        enemy_blinky_y = min(max(info["enemy_blinky_y"]-drift_y, 0), height)
    kp_ls = [[player_x, player_y], [enemy_sue_x, enemy_sue_y],
             [enemy_inky_x, enemy_inky_y], [enemy_pinky_x, enemy_pinky_y],
             [enemy_blinky_x, enemy_blinky_y], [fruit_x, fruit_y]]
    kp_ls = np.array(kp_ls)
    if image_size is not None:  # do not resize the image
        # NOTE: should only for raw_ram==False
        image_size = image_size - 1  # -1 to avoide point in edge cause patch position issue
        scale = np.array([[image_size/weight, image_size/height]] * kp_ls.shape[0])
        kp_ls = kp_ls * scale
    return kp_ls.astype('uint8')


def get_keypoints_frostbite(info, drift_x, drift_y, raw_ram, height=HEIGHT, weight=WEIGHT):
    if raw_ram:
        player_x = info["player_x"]
        player_y = info["player_y"]
        top_row_iceflow_x = info["top_row_iceflow_x"]
    else:
        # -1 to avoid point position in edge, which will cause error for patch position calculation
        weight = weight - 1
        height = height - 1
        player_x = min(max(info["player_x"]-drift_x, 0), weight)
        player_y = min(max(info["player_y"]-drift_y, 0), height)
        ice_flow_delt_x = 35
        ice_flow_delt_y = 25
        top_row_iceflow_x = min(max(info["top_row_iceflow_x"]-drift_x, 0), weight)
        top_row_iceflow_y = 100
        top_row_iceflow = np.array([[top_row_iceflow_x + ice_flow_delt_x * i, top_row_iceflow_y] for i in range(3)])

        second_row_iceflow_x = min(max(info["second_row_iceflow_x"]-drift_x, 0), weight)
        second_row_iceflow_y = top_row_iceflow_y + ice_flow_delt_y
        second_row_iceflow = np.array([[second_row_iceflow_x + ice_flow_delt_x * i, second_row_iceflow_y] for i in range(3)])

        third_row_iceflow_x = min(max(info["third_row_iceflow_x"]-drift_x, 0), weight)
        third_row_iceflow_y = second_row_iceflow_y + ice_flow_delt_y
        third_row_iceflow = np.array([[third_row_iceflow_x + ice_flow_delt_x * i, third_row_iceflow_y] for i in range(3)])

        fourth_row_iceflow_x = min(max(info["fourth_row_iceflow_x"]-drift_x, 0), weight)
        fourth_row_iceflow_y = third_row_iceflow_y + ice_flow_delt_y
        fourth_row_iceflow = np.array([[fourth_row_iceflow_x + ice_flow_delt_x * i, fourth_row_iceflow_y] for i in range(3)])

        iceflow = np.concatenate((top_row_iceflow, second_row_iceflow, third_row_iceflow, fourth_row_iceflow), axis=0)  # (3*4, 2)

    return np.concatenate((np.array([[player_x, player_y]]), iceflow), axis=0).astype('uint8')


def get_keypoints_pong(info, drift_x, drift_y, raw_ram, height=HEIGHT, weight=WEIGHT, image_size=None):
    # -1 to avoid point position in edge, which will cause error for patch position calculation
    if raw_ram:
        player_x = info["player_x"]
        player_y = info["player_y"]
        enemy_x = info["enemy_x"]
        enemy_y = info["enemy_y"]
        ball_x = info["ball_x"]
        ball_y = info["ball_y"]
    else:
        weight = weight - 1
        height = height - 1
        player_x = int(min(max(info["player_x"]-drift_x, 0), weight))
        player_y = int(min(max(info["player_y"]-drift_y, 0), height))
        enemy_x = int(min(max(info["enemy_x"]-drift_x, 0), weight))
        enemy_y = int(min(max(info["enemy_y"]-drift_y, 0), height))
        ball_x = int(min(max(info["ball_x"]-drift_x, 0), weight))
        ball_y = int(min(max(info["ball_y"]-drift_y, 0), height))
        if image_size is not None:  # do not resize the image
            image_size = image_size - 1  # -1 to avoide point in edge cause patch position issue
            # e.g. if image_size=80 and we have a point on (80, 80), patch_size=16,
            # then this point will be regarded as in patch (5, 5), however patch range should in [0, 4]
            player_x = player_x / weight * image_size
            player_y = player_y / height * image_size
            enemy_x = enemy_x / weight * image_size
            enemy_y = enemy_y / height * image_size
            ball_x = ball_x / weight * image_size
            ball_y = ball_y / height * image_size
    return np.array([[player_x, player_y], [enemy_x, enemy_y], [ball_x, ball_y]]).astype('uint8')


GET_KP_POS = {  # get KeyPoint POSition
    'pong': partial(get_keypoints_pong, drift_x=45, drift_y=15),
    # 'frostbite': partial(get_keypoints_frostbite, drift_x=0, drift_y=-40),
    'mspacman': partial(get_keypoints_mspacman, drift_x=10, drift_y=-10),
    'breakout': partial(get_keypoints_breakout, drift_x=37, drift_y=-10),
}

NUM_RAM_KPS = {
    'pong': 7,  # 1 for ball, 3*2 for two player
    # 'frostbite': 
    'mspacman': 6,
    'breakout': 2,
}