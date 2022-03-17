from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
import numpy as np
import time
from gym import spaces
import random
import re


class Adversarial_Robotarm(UnrealCv):
    def __init__(self, env, pose_range, cam_id=0, port=9000, targets=None,
                 ip='127.0.0.1', resolution=(160, 120)):
        self.arm = dict(
                pose=np.zeros(5),
                state=np.zeros(8),  # ground, left, left_in, right, right_in, body, reach
                grip=np.zeros(3),
                high=np.array(pose_range['high']),
                low=np.array(pose_range['low']),
        )
        super(Adversarial_Robotarm, self).__init__(env=env, port=port, ip=ip, cam_id=cam_id, resolution=resolution)

        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dic(self.targets)
        self.msgs_buffer = []

    def message_handler(self, msg):
        # msg: 'Hit object'
        self.msgs_buffer.append(msg)

    def read_message(self):
        msgs = self.msgs_buffer
        self.empty_msgs_buffer()
        return msgs

    def empty_msgs_buffer(self):
        self.msgs_buffer = []

    def set_arm_pose(self, pose):
        self.arm['pose'] = pose
        self.client.request('vset /arm/pose {} {} {} {} {}'.format(pose[0], pose[1], pose[2], pose[3], pose[4]))

    def move_arm(self, action):
        pose_tmp = self.arm['pose'] + action
        out_max = pose_tmp > self.arm['high']
        out_min = pose_tmp < self.arm['low']

        if out_max.sum() + out_min.sum() == 0:
            limit = False
        else:
            limit = True
            pose_tmp = out_max * self.arm['high'] + out_min * self.arm['low'] + ~(out_min + out_max) * pose_tmp
        self.set_arm_pose(pose_tmp)
        return limit

    def get_pose(self):
        # gym_unrealcv/envs/robotarm/interaction.py
        # --- not used ---
        res = None
        while res is None:
            res = self.client.request('vget /arm/pose')
        res = res.split()
        pose = [float(i) for i in res]
        pose = np.array(pose)
        self.arm['pose'] = pose
        return pose

    def get_tip_pose(self):
        cmd = 'vget /arm/tip_pose'
        result = None
        while result is None:
            result = self.client.request(cmd)
        pose = np.array([float(i) for i in result.split()])
        pose[1] = -pose[1]
        self.arm['grip'] = pose[:3]
        return pose

    def define_observation(self, cam_id, observation_type, setting, mode='direct'):
        if observation_type != 'Pose':
            state = self.get_observation(cam_id, observation_type, mode=mode)
        if observation_type == 'Color' or observation_type == 'CG':
            observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)  # for gym>=0.10
        elif observation_type == 'Depth':
            observation_space = spaces.Box(low=0, high=100, shape=state.shape, dtype=np.float16)  # for gym>=0.10
        elif observation_type == 'Rgbd':
            s_high = state
            s_high[:, :, -1] = 100.0  # max_depth
            s_high[:, :, :-1] = 255  # max_rgb
            s_low = np.zeros(state.shape)
            observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)  # for gym>=0.10`
        elif observation_type == 'Pose':
            s_high = setting['pose_range']['high'] + setting['goal_range']['high'] + setting['continuous_actions']['high'] + setting['camera_range']['high'] + setting['pose_xyz_range']['high'] # arm_pose, target_position, action
            s_low = setting['pose_range']['low'] + setting['goal_range']['low'] + setting['continuous_actions']['low'] + setting['camera_range']['low'] + setting['pose_xyz_range']['low']
            observation_space = spaces.Box(low=np.array(s_low), high=np.array(s_high))
        return observation_space

    def get_observation(self, cam_id, observation_type, target_pose=np.zeros(3), action=np.zeros(4), mode='direct'):
        if observation_type == 'Color':
            self.img_color = state = self.read_image(cam_id, 'lit', mode)
        elif observation_type == 'Depth':
            self.img_depth = state = self.read_depth(cam_id)
        elif observation_type == 'Rgbd':
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_depth = self.read_depth(cam_id)
            state = np.append(self.img_color, self.img_depth, axis=2)
        elif observation_type == 'Pose':
            self.target_pose = np.array(target_pose)
            self.get_tip_pose()
            state = np.concatenate((self.arm['pose'], self.target_pose, action, self.arm['grip']))
        return state

    def check_collision(self, obj='RobotArmActor_1'):
        'cmd : vget /arm/RobotArmActor_1/query collision'
        cmd = 'vget /arm/{obj}/query collision'
        res = self.client.request(cmd.format(obj=obj))
        if res == 'true':
            return True
        else:
            return False

    def get_keypoints_2d(self, joint_dir, vertex_seq, actor_name="RobotArmActor_3"):
        tmp_joint_dataset = vdb.vdb.Dataset(joint_dir)

        res = None
        while res is None:
            res = self.client.request('vget /arm/keypoints %s' % joint_dir)
        res = res.split()
        vertex_2d = [float(i) for i in res]
        vertex_2d = np.array(vertex_2d)
        vertex_2d = vertex_2d.reshape((-1, 3))

        cam_info = self.get_cam_info()

        loc = cam_info['Location']
        rot = cam_info['Rotation']

        cam = vdb.d3.CameraPose(loc['X'], loc['Y'], loc['Z'],
                                rot['Pitch'], rot['Yaw'], rot['Roll'],
                                cam_info['FilmWidth'], cam_info['FilmHeight'], cam_info['FilmWidth'] / 2)

        vertex_2d = cam.project_to_2d(vertex_2d)

        ids = tmp_joint_dataset.get_ids_with_joints()[-1:]
        joint_2d = get_joint_2d_raw(tmp_joint_dataset, ids, cam, actor_name)
        joint_2d = joint_2d[1:]

        num_vertex = len(vertex_seq)
        pts = np.zeros((num_vertex, 2))
        for i in range(num_vertex):
            pts[i] = np.average(vertex_2d[vertex_seq[i]], axis=0)
        pts = np.concatenate((joint_2d, pts), axis=0)

        self.arm['keypoints'] = pts
        return pts

    def get_cam_info(self):
        cam_info = {}
        res = None
        while res is None:
            res = self.client.request('vget /camera/1/info')
        res = res.split()
        res = [float(i) for i in res]

        cam_info['Location'] = {'X': res[0], 'Y': res[1], 'Z': res[2]}
        cam_info['Rotation'] = {'Pitch': res[3], 'Yaw': res[4], 'Roll': res[5]}
        cam_info['FilmWidth'] = int(res[6])
        cam_info['FilmHeight'] = int(res[7])
        cam_info['Fov'] = res[8]

        return cam_info

    def get_seg(self):
        view_mode = 'seg'
        res = None
        while res is None:
            res = self.client.request('vget /camera/1/{} png'.format(view_mode))
        image_rgb = self.decode_png(res)
        image_rgb = image_rgb[:, :, :-1]  # delete alpha channel
        return image_rgb

    def set_size(self, width, height):
        res = None
        while res is None:
            res = self.client.request('vset /camera/1/size %d %d' % (width, height))

    def save_img(self, data_dir):
        self.client.request('vset /data_capture/capture_frame ' + data_dir)

    def set_camera_pose(self):
        # randomize camera position and fix
        dist = np.random.uniform(500, 800)
        pitch = -np.random.uniform(5, 45)
        yaw = np.random.uniform(0, 360)
        roll = np.random.uniform(0, 0)
        res = None
        while res is None:
            res = self.client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**locals()))
        x = int(-dist * cos(pitch * pi / 180) * cos(yaw * pi / 180))
        y = int(-dist * cos(pitch * pi / 180) * sin(yaw * pi / 180))
        z = int(-dist * sin(pitch * pi / 180))

        res = None
        while res is None:
            res = self.client.request('vset /camera/1/location {x} {y} {z}'.format(**locals()))

        return np.array([x, y, z, pitch, yaw, roll])
