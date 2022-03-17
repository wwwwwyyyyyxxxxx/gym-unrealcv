import math
import os
import random
import time
import torch
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.utils import env_unreal, misc
from gym_unrealcv.envs.adversarial_robotarm.interaction import Adversarial_Robotarm
from pose.models.integral import Integral
from pose.utils.imutils import draw_labelmap, to_torch
from pose.utils.transforms import transform
import json


class UnrealCvAdversarial_RobotArm_reach(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type='keyboard',    # keyboard, bp
                 action_type='continuous',   # 'discrete', 'continuous'
                 observation_type='Pose',  # 'color', 'depth', 'rgbd' . 'pose'
                 version=0,  # train, test
                 docker=False,
                 resolution=(80, 80),
                 ):

        # load and process setting
        setting = misc.load_env_setting(setting_file)
        self.cam_id = setting['cam_view_id']
        self.max_steps = setting['maxsteps']
        self.camera_pose = setting['camera_pose']
        self.discrete_actions = setting['discrete_actions']
        self.continuous_actions = setting['continuous_actions']
        self.pose_range = setting['pose_range']
        self.goal_range = setting['goal_range']
        self.env_bin = setting['env_bin']
        self.env_map = setting['env_map']
        self.objects = setting['objects']
        self.docker = docker
        self.reset_type = reset_type
        self.resolution = resolution
        self.version = version
        self.launched = False
        self.is_train = True

        # define action type
        self.action_type = action_type
        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'Continuous':
            self.action_space = spaces.Box(low=np.array(self.continuous_actions['low']),
                                           high=np.array(self.continuous_actions['high']))

        self.pose_low = np.array(self.pose_range['low'])
        self.pose_high = np.array(self.pose_range['high'])

        # Dynamics: delta_pose = action*mult, where mult~[lower, upper]
        self.lower_mult = setting['lower_mult']
        self.upper_mult = setting['upper_mult']

        self.pose_dim = 5

        self.count_steps = 0
        self.count_eps = 0

        # define observation space,
        # color, depth, rgbd...
        self.launch_env()
        self.observation_type = observation_type
        self.observation_space = self.unrealcv.define_observation(self.cam_id, observation_type, setting)

    def launch_env(self):
        if self.launched:
            return True
        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=self.env_bin, ENV_MAP=self.env_map)
        env_ip, env_port = self.unreal.start(self.docker, self.resolution)

        # connect UnrealCV
        self.unrealcv = Adversarial_Robotarm(cam_id=self.cam_id,
                                             pose_range=self.pose_range,
                                             port=env_port,
                                             ip=env_ip,
                                             targets=[],
                                             env=self.unreal.path2env,
                                             resolution=self.resolution)
        self.launched = True
        return self.launched

    def step(self, action):
        info = dict(
            Done=False,
            Reward=0.0,
            Action=action,
            Steps=self.count_steps,
            TargetPose=self.goal_pos_trz,
            Color=None,
            Depth=None,
        )
        action = np.squeeze(action)
        self.count_steps += 1
        done = False

        # take a action
        if self.action_type == 'Discrete':
            action = self.discrete_actions[action]
        elif self.action_type == 'Continuous':
            action = np.append(action, 0)
        if self.need_save_img:
            action = self.calc_action(action)
        arm_state = self.unrealcv.move_arm(action)

        tip_pose = self.unrealcv.get_tip_pose()
        distance_xyz = self.get_distance(self.goal_pos_xyz, tip_pose)
        collision = self.unrealcv.check_collision()  # check collision

        is_out_of_cam, keypoints_2d = self.out_of_cam()

        # reward function
        reward = - 0.01 * distance_xyz

        info['success'] = False
        if arm_state or collision or is_out_of_cam:  # reach limitation or collision
            done = True
            reward = -10
        elif distance_xyz < 20:  # reach
            reward = 1 - 0.1 * distance_xyz
            self.count_reach += 1
            if self.count_reach >= self.count_th:
                done = True
                reward = (1 - 0.05 * distance_xyz) * 100
                info['success'] = True

        estimator_reward = 0
        if not is_out_of_cam:
            estimator_reward = float(self.estimator_reward(image, action, pose=pose, keypoints_2d=keypoints_2d))
        reward += estimator_reward

        # Get observation
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, self.goal_pos_trz, action)
        info['Done'] = done
        info['Reward'] = reward
        info['Estimator_Reward'] = estimator_reward

        if self.need_save_img:
            self.unrealcv.save_img(self.data_dir)

        return state, reward, done, info

    def seed(self, seed=None):
        pass

    def reset(self):
        self.launch_env()
        self.count_eps += 1

        while True:
            if self.is_train or self.need_save_img:
                # for training
                init_pose = [random.uniform(-90, 90),
                             random.uniform(-15, 15),
                             random.uniform(-30, 30),
                             random.uniform(-30, 30),
                             0]
                self.goal_pos_trz = self.sample_goal(-1)
                self.count_th = 5
            else:
                # for testing
                init_pose = [0, 0, 0, 0, 0]
                self.goal_pos_trz = self.sample_goal(self.count_eps)
                self.count_th = 3
            self.unrealcv.set_arm_pose(init_pose)

            tip_z = self.unrealcv.get_tip_pose()[2]
            if tip_z < 0:
                continue

            is_out_of_cam, _ = self.out_of_cam()
            if not is_out_of_cam:
                break

        self.goal_pos_xyz = self.trz2xyz(self.goal_pos_trz)
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, self.goal_pos_trz)

        self.count_steps = 0
        self.count_reach = 0
        self.unrealcv.set_obj_location(self.objects[0], [0, 0, -50])
        self.unrealcv.set_obj_rotation(self.objects[0], [0, 0, 0])
        self.arm_pose_last = self.unrealcv.get_arm_pose('new')
        self.unrealcv.empty_msgs_buffer()

        if self.need_save_img:
            self.data_dir = os.path.join(self.data_dir_root, "seq%05d" % self.count_eps)
            self.unrealcv.save_img(self.data_dir)

        return state

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close:
            self.unreal.close()
        return self.unrealcv.img_color

    def get_distance(self, target, current, norm=False, n=3):
        error = np.array(target[:n]) - np.array(current[:n])
        if norm:
            error = error/np.array(self.goal_range['high'])
        distance = np.linalg.norm(error)
        return distance

    def xyz2trz(self, xyz):
        theta = np.arctan2(xyz[0], xyz[1])/np.pi*180
        r = np.linalg.norm(xyz[:2])
        z = xyz[2]
        return np.array([theta, r, z])

    def trz2xyz(self, trz):
        x = np.sin(trz[0]/180.0*np.pi)*trz[1]
        y = np.cos(trz[0]/180.0*np.pi)*trz[1]
        z = trz[2]
        return np.array([x, y, z])

    def sample_goal(self, count_eps):
        if count_eps < 0:
            theta = random.uniform(self.goal_range['low'][0], self.goal_range['high'][0])
            r = random.uniform(self.goal_range['low'][1], self.goal_range['high'][1])
            z = random.uniform(self.goal_range['low'][2], min(r, self.goal_range['high'][2]))
            goal = np.array([theta, r, z])
        else:
            yaws = [0, -45, 45]
            length = [150, 200, 250]
            heights = [20, 20, 20]
            goal = np.array([yaws[int(self.count_eps / 3 % 3)],
                             length[int(self.count_eps % 3)],
                             heights[int(self.count_eps % 3)]])
        return goal

    def set_mode(self, mode, data_dir_root=None, start_eps=0):
        self.count_eps = 0
        if mode == "train":
            self.is_train = True
            self.need_save_img = False
        elif mode == "data_generation":
            self.is_train = True
            self.need_save_img = True
            assert data_dir_root is not None
            self.data_dir_root = data_dir_root
            self.count_eps = start_eps
        elif mode == "test":
            self.is_train = False
            self.need_save_img = False
        else:
            raise NotImplementedError

    def construct_pose_estimator(self, pose_estimator_class, need_seq, **kwargs):
        self.need_seq = need_seq
        with open(os.path.join(kwargs['meta_dir'], 'vertex.json'), 'r') as f:
            self.vertex_seq = json.load(f)
        if pose_estimator_class is None:
            self.pose_estimator = None
            return
        self.pose_estimator = pose_estimator_class(**kwargs)
        if self.anno_type == 'heatmap' or self.anno_type == '2d':
            self.sigma = 1
            self.label_type = "Gaussian"
            if self.anno_type == '2d':
                self.integral = Integral()
        self.lambda_pose_estimator = 16

    def estimator_reward(self, image, action, pose=None, keypoints_2d=None):
        if self.pose_estimator is None:
            return 0
        if self.anno_type == '3d':
            if self.need_seq:
                output = self.pose_estimator.get_pose(image, action)
            else:
                output = self.pose_estimator.get_pose(image)
            if len(output) == 4:
                output = np.insert(output, 4, 0)
            return F.mse_loss(torch.tensor(output), torch.tensor(pose), reduction='mean') / self.lambda_pose_estimator
        elif self.anno_type == 'heatmap' or self.anno_type == '2d':
            if self.need_seq:
                score_map = self.pose_estimator.get_score_map(image, action)
            else:
                score_map = self.pose_estimator.get_score_map(image)
            if self.anno_type == '2d':
                score_map, = self.integral(score_map)
            target = keypoints_2d
            target = self.get_target(target)
            return F.mse_loss(score_map, target, reduction='mean') / self.lambda_pose_estimator
        else:
            return 0

    def out_of_cam(self):
        keypoints_2d = self.unrealcv.get_keypoints_2d(self.tmp_joint_dir, self.vertex_seq, self.actor_name)
        return (keypoints_2d < 0).any(), keypoints_2d

    def get_target(self, pts, sigma=1, label_type="Gaussian", out_res=64):
        color = [127, 0, 0]

        scale_factor = 60.0

        tpts = pts.copy()

        nparts = pts.shape[0]

        segmask = self.unrealcv.get_seg()

        binary_arm = vdb.get_obj_mask(segmask, color)
        bb = vdb.seg2bb(binary_arm)
        x0, x1, y0, y1 = bb

        c = np.array([(x0 + x1), (y0 + y1)]) / 2
        s = np.sqrt((y1 - y0) * (x1 - x0)) / scale_factor

        c = c + np.array([-30 + 60 * np.random.random(), -30 + 60 * np.random.random()])  # random move
        s *= 0.6 * (1 + 2 * np.random.random())  # random scale

        rf = 15
        r = -rf + 2 * np.random.random() * rf  # random rotation
        # r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

        target = torch.zeros(nparts, out_res, out_res)
        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], c, s, [out_res, out_res], rot=r))
                if self.anno_type == 'heatmap':
                    target[i] = draw_labelmap(target[i], tpts[i], sigma, type=label_type)

        if self.anno_type == 'heatmap':
            return target
        elif self.anno_type == '2d':
            target = torch.from_numpy(tpts).float().clone().detach()
            return target
        else:
            return None

    def calc_action(self, action):
        mult = np.random.uniform(self.lower_mult, self.upper_mult, action.shape)
        action = action * (0.01 * mult)
        return action
