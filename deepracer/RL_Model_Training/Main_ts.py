"""
Environments that uses TS for deepracer.
- Includes appropriate functions to make the environment working with baselines code.
- Credits:
The environment code is taken from the repo of deepracer with modifications.
The model loading code is taken from open AI baselines with modifications done to allow for timing characteristics.


Setting:
- Latency is varied between: 10ms to 120ms.
- A fixed sampling interval of 33ms is used when latency is less than the sampling interval,
and for cases when latency is greater than the sampling interval, both latency and sampling interval values are matched.
"""
log_path = 'Path_to_save_checkpoints'
global_seed = 0


import os
G_evaluation_env = None
G_num_episodes_evaluation = 1

import random

import tensorflow as tf

tf.set_random_seed(global_seed)

import numpy as np

np.random.seed(global_seed)

random.seed(global_seed)

nsteps_size = 7000
nminibatches_size = 50


import sys
print(sys.executable)

import sys
#print(sys.path)
del_path = []
for p in reversed(sys.path):
    if 'python2.7' in p:
        sys.path.remove(p)
        del_path.append(p)
#print(sys.path)
import cv2
for p in del_path:
    sys.path.append(p)

import gym
import markov
import markov.environments
import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args#, #make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

#######################################################Randomizations
import sys
#print(sys.path)
del_path = []
for p in reversed(sys.path):
    if 'python2.7' in p:
        sys.path.remove(p)
        del_path.append(p)
#print(sys.path)
import cv2
for p in del_path:
    sys.path.append(p)
#print(sys.path)

import numpy as np

def random_hue(x, saturation=None):
    if saturation is None:
        saturation = np.random.randint(10)

    x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    v = x[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    x[:, :, 2] = v
    x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)

    return x

def random_saturation(x, saturation=None):
    if saturation is None:
        saturation = np.random.randint(30)

    x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    v = x[:, :, 2]
    v = np.where(v <= 255 - saturation, v - saturation, 255)
    x[:, :, 2] = v
    x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)

    return x

def random_brightness(x, brightness=None):
    if brightness is None:
        brightness = np.random.uniform(0.5, 2.5)

    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return x

def random_contrast(x, contrast=None):
    if contrast is None:
        contrast = np.random.randint(50)  # too large will be too dark

    x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    x[:, :, 2] = [[max(pixel - contrast, 0) if pixel < 190 else min(
        pixel + contrast, 255)
                   for pixel in row] for row in x[:, :, 2]]
    x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)
    return x


def random_color(x):

    #t1 = time.time()
    x = random_hue(x)

    #t2 = time.time()
    #print('Time random_hue:',(t2-t1)*1000)

    #t1 = time.time()
    x = random_saturation(x)
    #t2 = time.time()
    #print('Time random_saturation:',(t2-t1)*1000)

    #t1 = time.time()
    x = random_brightness(x)
    #t2 = time.time()
    #print('Time random_brightness:',(t2-t1)*1000)


    #t1 = time.time()
    #x = random_contrast(x)
    #t2 = time.time()
    #print('Time random_contrast:',(t2-t1)*1000)


    return x

def trans(x, trans_range):
    H, W, nc = x.shape

    # Translation
    tr_y = trans_range * (np.random.uniform() - 0.5)
    tr_x = 20 * (np.random.uniform() - .5)
    Trans_M = np.float32([[1, 0, tr_x],
                          [0, 1, tr_y]])
    x = cv2.warpAffine(x, Trans_M, (W, H))

    return x

def shadow(x):
    H, W, _ = x.shape

    top_x, bot_x = 0, W
    top_y, bot_y = H * np.random.uniform(), H * np.random.uniform()

    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    shadow_mask = 0 * hsv[:, :, 1]
    X_m, Y_m = np.mgrid[0:H, 0:W]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (
                Y_m - top_y) >= 0)] = 1

    shadow_density = .5
    left_side = shadow_mask == 1
    right_side = shadow_mask == 0

    if np.random.randint(2) == 1:
        hsv[:, :, 2][left_side] = hsv[:, :, 2][left_side] * shadow_density
    else:
        hsv[:, :, 2][right_side] = hsv[:, :, 2][right_side] * shadow_density

    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return x

def sharpen(x):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    x = cv2.filter2D(x, -1, kernel)
    return x

def salt_and_pepper(x, p=0.5, a=0.009):
    noisy = x.copy()

    # salt
    num_salt = np.ceil(a * x.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in x.shape]
    noisy[tuple(coords)] = 1

    # pepper
    num_pepper = np.ceil(a * x.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in x.shape]
    noisy[tuple(coords)] = 0

    return noisy


def do_randomization(observation ,p=0.8):

    #t1 = time.time()
    if np.random.uniform() > p:
        observation = random_color(observation)

    #t2 = time.time()
    #print('Time random_color:',(t2-t1)*1000)

    #t1 = time.time()
    if np.random.uniform() > p:
        observation = shadow(observation)

    #t2 = time.time()
    #print('Time shadow:',(t2-t1)*1000)


    #t1 = time.time()
    if np.random.uniform() > p:
        observation = sharpen(observation)
    #t2 = time.time()
    #print('Time sharpen:',(t2-t1)*1000)


    #t1 = time.time()
    if np.random.uniform() > p:
        observation = salt_and_pepper(observation)
    #t2 = time.time()
    #print('Time salt_and_pepper:',(t2-t1)*1000)


    return observation
#######################################################End Randomizations


# Changing the sampling and latency input to the model


import time

import gym
import queue

import numpy as np
from gym import spaces
from PIL import Image
import os
import math
from rotation import Rotation
from collections import OrderedDict
import random
import bisect
import json
import math

# Type of worker
SIMULATION_WORKER = "SIMULATION_WORKER"
SAGEMAKER_TRAINING_WORKER = "SAGEMAKER_TRAINING_WORKER"

node_type = os.environ.get("NODE_TYPE", SIMULATION_WORKER)

#saving the debug data
import pickle

if node_type == SIMULATION_WORKER:
    import rospy
    from ackermann_msgs.msg import AckermannDriveStamped
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import GetLinkState, GetModelState, JointRequest
    from gazebo_msgs.srv import SetModelState
    from std_msgs.msg import Float64

    from sensor_msgs.msg import Image as sensor_image
    from deepracer_msgs.msg import Progress

    from shapely.geometry import Point, Polygon
    from shapely.geometry.polygon import LinearRing, LineString

TRAINING_IMAGE_SIZE = (160, 120)
FINISH_LINE = 1000

# REWARD ENUM
CRASHED = -30.0
NO_PROGRESS = -1
FINISHED = 10000000.0
MAX_STEPS = 100000000

# WORLD NAME
EASY_TRACK_WORLD = 'easy_track'
MEDIUM_TRACK_WORLD = 'medium_track'
HARD_TRACK_WORLD = 'hard_track'


# Normalized track distance to move with each reset
ROUND_ROBIN_ADVANCE_DIST = 0.02#0.02 #0.01

# List of required velocity topics, one topic per wheel
VELOCITY_TOPICS = ['/racecar/left_rear_wheel_velocity_controller/command',
                   '/racecar/right_rear_wheel_velocity_controller/command',
                   '/racecar/left_front_wheel_velocity_controller/command',
                   '/racecar/right_front_wheel_velocity_controller/command']


# List of required steering hinges
STEERING_TOPICS = ['/racecar/left_steering_hinge_position_controller/command',
                   '/racecar/right_steering_hinge_position_controller/command']

# List of all effort joints
EFFORT_JOINTS = ['/racecar/left_rear_wheel_joint', '/racecar/right_rear_wheel_joint',
                 '/racecar/left_front_wheel_joint','/racecar/right_front_wheel_joint',
                 '/racecar/left_steering_hinge_joint','/racecar/right_steering_hinge_joint']

# Radius of the wheels of the car in meters
WHEEL_RADIUS = 0.1

# Size of the image queue buffer, we want this to be one so that we consume 1 image
# at a time, but may want to change this as we add more algorithms
IMG_QUEUE_BUF_SIZE = 1


#print(delays_array)

### Gym Env ###
class DeepRacerEnv(gym.Env):
    def __init__(self):

        self.sampling_rate = 30.0

        self.sampling_sleep = (1.0/self.sampling_rate)

        self.sampling_rates = [30.0, 30.0]
        self.sampling_rate_index = 0

        self.latencies = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]

        self.latency_index = 0

        self.latency_max_num_steps = 500 # for these steps latency will be fixed or change on reset or done after 250.

        self.latency_steps = 0

        self.latency = 10.0 #10 is the starting latency

        self.model_running_time = (2.0/1000.0) #model runtime


        screen_height = TRAINING_IMAGE_SIZE[1]
        screen_width = TRAINING_IMAGE_SIZE[0]


        self.on_track = 0
        self.progress = 0
        self.yaw = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.distance_from_center = 0
        self.distance_from_border_1 = 0
        self.distance_from_border_2 = 0
        self.steps = 0
        self.progress_at_beginning_of_race = 0

        self.reverse_dir = False
        self.start_ndist = 0.0

        # actions -> steering angle, throttle
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32)

        # given image from simulator
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(screen_height, screen_width, 1), dtype=np.uint8)

        self.allow_servo_step_signals = True

        #stores the time when camera images are received
        self.cam_update_time=[]

        #stores the time when consequetive actions are send
        self.cons_action_send_time=[]

        #stores the time when progress updates are received
        self.progress_update_time = []

        #folder location to store the debug data
        self.debug_data_folder = []

        self.debug_index = 0


        if node_type == SIMULATION_WORKER:
            # ROS initialization
            rospy.init_node('rl_coach', anonymous=True)


            self.ack_publisher = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',
                                                 AckermannDriveStamped, queue_size=100)
            self.racecar_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

            self.clear_forces_client = rospy.ServiceProxy('/gazebo/clear_joint_forces',
                                                          JointRequest)


            # Subscribe to ROS topics and register callbacks
            rospy.Subscriber('/progress', Progress, self.callback_progress)
            rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self.callback_image)

            self.world_name = 'hard_track'#rospy.get_param('WORLD_NAME')
            self.set_waypoints()

        waypoints = self.waypoints
        is_loop = np.all(waypoints[0,:] == waypoints[-1,:])
        if is_loop:
            self.center_line = LinearRing(waypoints[:,0:2])

        else:
            self.center_line = LineString(waypoints[:,0:2])

        self.center_dists = [self.center_line.project(Point(p), normalized=True) for p in self.center_line.coords[:-1]] + [1.0]
        self.track_length = self.center_line.length



        self.reward_in_episode = 0
        self.prev_progress = 0
        self.steps = 0

        # Create the publishers for sending speed and steering info to the car
        self.velocity_pub_dict = OrderedDict()
        self.steering_pub_dict = OrderedDict()

        for topic in VELOCITY_TOPICS:
            self.velocity_pub_dict[topic] = rospy.Publisher(topic, Float64, queue_size=1)

        for topic in STEERING_TOPICS:
            self.steering_pub_dict[topic] = rospy.Publisher(topic, Float64, queue_size=1)

    def get_data_debug(self):
        print("center_line",self.center_line)
        print("track_length",self.track_length)

    def reset(self,inp_x=1.75,inp_y=0.6):
        if node_type == SAGEMAKER_TRAINING_WORKER:
            return self.observation_space.sample()
        #print('Total Reward Reward=%.2f' % self.reward_in_episode,
        #      'Total Steps=%.2f' % self.steps)
        #self.send_reward_to_cloudwatch(self.reward_in_episode)

        self.reward_in_episode = 0
        self.reward = None
        self.done = False
        self.next_state = None
        self.image = None
        self.steps = 0
        self.prev_progress = 0

        # Reset car in Gazebo
        self.send_action(0, 0)  # set the throttle to 0


        self.racecar_reset(0, 0)

        self.infer_reward_state(0, 0)


        self.cam_update_time = []
        self.cons_action_send_time = []
        self.progress_update_time = []
        self.debug_index= self.debug_index+1

        return self.next_state


    def add_latency_to_image(self,observation):

        observation = observation.reshape(observation.shape[0],observation.shape[1],1)

        #print('Set latency is:',self.latency*self.latency_max)
        observation[119, 159, 0] = int(self.latency)

        #setting the sampling rate
        observation[119, 158, 0] = int(self.sampling_rate)

        #print(observation[119, 159, 0],observation[119, 158, 0] )

        return observation

    def convert_rgb_to_gray(self, observation):
        r, g, b = observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]
        observation = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return observation





    def set_next_state(self):
        if(self.image!=None):

            #t1 = time.time()

            image_data = self.image

            # Read the image and resize to get the state
            #print(image_data.width, image_data.height)
            image = Image.frombytes('RGB', (image_data.width, image_data.height), image_data.data, 'raw', 'RGB', 0, 1)
            image = image.resize(TRAINING_IMAGE_SIZE, resample=2)
            image = np.array(image)

            image = do_randomization(image)

            image = self.convert_rgb_to_gray(image)
            image = self.add_latency_to_image(image)

            self.next_state = image



    def racecar_reset(self, ndist, next_index):
        rospy.wait_for_service('gazebo/set_model_state')

        #random_start = random.random()

        prev_index, next_index = self.find_prev_next_waypoints(self.start_ndist)

        # Compute the starting position and heading
        #start_point = self.center_line.interpolate(ndist, normalized=True)
        start_point = self.center_line.interpolate(self.start_ndist, normalized=True)
        start_yaw = math.atan2(self.center_line.coords[next_index][1] - start_point.y,
                               self.center_line.coords[next_index][0] - start_point.x)
        start_quaternion = Rotation.from_euler('zyx', [start_yaw, 0, 0]).as_quat()

        # Construct the model state and send to Gazebo
        model_state = ModelState()
        model_state.model_name = 'racecar'
        model_state.pose.position.x = start_point.x
        model_state.pose.position.y = start_point.y
        model_state.pose.position.z = 0
        model_state.pose.orientation.x = start_quaternion[0]
        model_state.pose.orientation.y = start_quaternion[1]
        model_state.pose.orientation.z = start_quaternion[2]
        model_state.pose.orientation.w = start_quaternion[3]
        model_state.twist.linear.x = 0
        model_state.twist.linear.y = 0
        model_state.twist.linear.z = 0
        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = 0

        self.racecar_service(model_state)

        for joint in EFFORT_JOINTS:
            self.clear_forces_client(joint)


        #keeping track where to start the car
        self.reverse_dir = not self.reverse_dir
        self.start_ndist = (self.start_ndist + ROUND_ROBIN_ADVANCE_DIST) % 1.0

        self.progress_at_beginning_of_race = self.progress



    def find_prev_next_waypoints(self, ndist):
        if self.reverse_dir:
            next_index = bisect.bisect_left(self.center_dists, ndist) - 1
            prev_index = next_index + 1
            if next_index == -1: next_index = len(self.center_dists) - 1
        else:
            next_index = bisect.bisect_right(self.center_dists, ndist)
            prev_index = next_index - 1
            if next_index == len(self.center_dists): next_index = 0
        return prev_index, next_index


    def step(self, action):

        self.latency_steps = self.latency_steps+1

        #print('latency set in env:',self.latency)

        #bookeeping when the action was send
        #self.cons_action_send_time.append([self.steps,time.time()])

        latency = (self.latency-2.0)/1000.0
        #10ms latency is substracted, because that is the avg default latency observed on the training machine

        if latency>0.001:
            time.sleep(latency)


        else:
            latency = 0.0


        # Initialize next state, reward, done flag
        self.next_state = None
        self.reward = None
        self.done = False

        # Send this action to Gazebo and increment the step count
        self.steering_angle = float(action[0])
        self.speed = float(action[1])

        self.send_action(self.steering_angle, self.speed)
        self.steps += 1

        #sleep to control sampling rate
        to_sleep = (self.sampling_sleep - self.model_running_time - latency)



        if to_sleep>0.001:
            time.sleep(to_sleep)


        if self.latency_steps == self.latency_max_num_steps:

            #update the latency
            self.latency_index = (self.latency_index+1) % (len(self.latencies))
            self.latency = self.latencies[self.latency_index]


            #update the sampling rate
            self.sampling_rate_index  = random.randint(0,1)
            self.sampling_rate = self.sampling_rates[self.sampling_rate_index]

            self.sampling_sleep = (1.0/self.sampling_rate)

            if (self.latency/1000.0)> self.sampling_sleep: # match sampling input to the model and latency
                 self.sampling_rate = 1000.0/self.latency


            self.latency_steps = 0


        # Compute the next state and reward
        self.infer_reward_state(self.steering_angle, self.speed)


        return self.next_state, self.reward, self.done, {}


    def send_action(self, steering_angle, speed):
        # Simple v/r to computes the desired rpm
        wheel_rpm = speed/WHEEL_RADIUS

        for _, pub in self.velocity_pub_dict.items():
            pub.publish(wheel_rpm)

        for _, pub in self.steering_pub_dict.items():
            pub.publish(steering_angle)


    def callback_image(self, data):
        self.image = data

        #bookeeping when the image was received
        #self.cam_update_time.append([self.steps,time.time()])


    def callback_progress(self, data):
        self.on_track = not (data.off_track)
        self.progress = data.progress
        self.yaw = data.yaw
        self.x = data.x
        self.y = data.y
        self.z = data.z
        self.distance_from_center = data.distance_from_center
        self.distance_from_border_1 = data.distance_from_border_1
        self.distance_from_border_2 = data.distance_from_border_2

        #bookeeping when the progress was received
        #self.progress_update_time.append([self.steps,time.time()])




    def reward_function (self, on_track, x, y, distance_from_center,
    throttle, steering, track_width):

        marker_1 = 0.1 * track_width
        marker_2 = 0.15 * track_width
        marker_3 = 0.20 * track_width

        reward = (track_width - distance_from_center) #max reward = 0.44


        if distance_from_center >= 0.0 and distance_from_center <= marker_1:
            reward = reward * 2.5 #0.90, 0.44 max is scaled to 1.0
        elif distance_from_center <= marker_2:
            reward = reward * 1.33 #0.85, 0.375 max is scaled to 0.5
        elif distance_from_center <= marker_3:
            reward = reward * 0.71  #0.80, 0.352 max is scaled to 0.25
        else:
            reward = 0.001  # may go close to off track

        # penalize reward for the car taking slow actions

        if throttle < 1.6 and reward>0:
            reward *= 0.95

        if throttle < 1.4 and reward>0:
            reward *= 0.95


        return float(reward)


    def infer_reward_state(self, steering_angle, throttle):

        #state has to be set first, because we need most accurate reward signal
        self.set_next_state()


        on_track = self.on_track

        done = False

        if on_track != 1:
            reward = CRASHED
            done = True

        else:
            reward = self.reward_function(on_track, self.x, self.y, self.distance_from_center,
                                          throttle, steering_angle, self.road_width)


        #after 500 steps in episode we want to restart it
        if self.steps==500:
            done = True

            if reward > 0: #car is not crashed
                reward = reward *5.0 #bonus on completing 500 steps


        self.reward_in_episode += reward
        self.reward = reward
        self.done = done






    def set_waypoints(self):
        if self.world_name.startswith(MEDIUM_TRACK_WORLD):
            self.waypoints = vertices = np.zeros((8, 2))
            self.road_width = 0.50
            vertices[0][0] = -0.99; vertices[0][1] = 2.25;
            vertices[1][0] = 0.69;  vertices[1][1] = 2.26;
            vertices[2][0] = 1.37;  vertices[2][1] = 1.67;
            vertices[3][0] = 1.48;  vertices[3][1] = -1.54;
            vertices[4][0] = 0.81;  vertices[4][1] = -2.44;
            vertices[5][0] = -1.25; vertices[5][1] = -2.30;
            vertices[6][0] = -1.67; vertices[6][1] = -1.64;
            vertices[7][0] = -1.73; vertices[7][1] = 1.63;
        elif self.world_name.startswith(EASY_TRACK_WORLD):
            self.waypoints = vertices = np.zeros((2, 2))
            self.road_width = 0.90
            vertices[0][0] = -1.08;   vertices[0][1] = -0.05;
            vertices[1][0] =  1.08;   vertices[1][1] = -0.05;
        else:
            self.waypoints = vertices = np.zeros((30, 2))
            self.road_width = 0.44
            vertices[0][0] = 1.5;     vertices[0][1] = 0.58;
            vertices[1][0] = 5.5;     vertices[1][1] = 0.58;
            vertices[2][0] = 5.6;     vertices[2][1] = 0.6;
            vertices[3][0] = 5.7;     vertices[3][1] = 0.65;
            vertices[4][0] = 5.8;     vertices[4][1] = 0.7;
            vertices[5][0] = 5.9;     vertices[5][1] = 0.8;
            vertices[6][0] = 6.0;     vertices[6][1] = 0.9;
            vertices[7][0] = 6.08;    vertices[7][1] = 1.1;
            vertices[8][0] = 6.1;     vertices[8][1] = 1.2;
            vertices[9][0] = 6.1;     vertices[9][1] = 1.3;
            vertices[10][0] = 6.1;    vertices[10][1] = 1.4;
            vertices[11][0] = 6.07;   vertices[11][1] = 1.5;
            vertices[12][0] = 6.05;   vertices[12][1] = 1.6;
            vertices[13][0] = 6;      vertices[13][1] = 1.7;
            vertices[14][0] = 5.9;    vertices[14][1] = 1.8;
            vertices[15][0] = 5.75;   vertices[15][1] = 1.9;
            vertices[16][0] = 5.6;    vertices[16][1] = 2.0;
            vertices[17][0] = 4.2;    vertices[17][1] = 2.02;
            vertices[18][0] = 4;      vertices[18][1] = 2.1;
            vertices[19][0] = 2.6;    vertices[19][1] = 3.92;
            vertices[20][0] = 2.4;    vertices[20][1] = 4;
            vertices[21][0] = 1.2;    vertices[21][1] = 3.95;
            vertices[22][0] = 1.1;    vertices[22][1] = 3.92;
            vertices[23][0] = 1;      vertices[23][1] = 3.88;
            vertices[24][0] = 0.8;    vertices[24][1] = 3.72;
            vertices[25][0] = 0.6;    vertices[25][1] = 3.4;
            vertices[26][0] = 0.58;   vertices[26][1] = 3.3;
            vertices[27][0] = 0.57;   vertices[27][1] = 3.2;
            vertices[28][0] = 1;      vertices[28][1] = 1;
            vertices[29][0] = 1.25;   vertices[29][1] = 0.7;

    def get_closest_waypoint(self):
        res = 0
        index = 0
        x = self.x
        y = self.y
        minDistance = float('inf')
        for row in self.waypoints:
            distance = math.sqrt((row[0] - x) * (row[0] - x) + (row[1] - y) * (row[1] - y))
            if distance < minDistance:
                minDistance = distance
                res = index
            index = index + 1
        return res

class DeepRacerDiscreteEnv(DeepRacerEnv):
    def __init__(self):
        DeepRacerEnv.__init__(self)

        with open('custom_files/model_metadata.json', 'r') as f:
            model_metadata = json.load(f)
            self.json_actions = model_metadata['action_space']

        self.action_space = spaces.Discrete(len(self.json_actions))

        print("Intialized action space")
        print(self.json_actions)
        print("num of actions",self.action_space )

    def step(self, action):

        action = int(action)
        # Convert discrete to continuous
        steering_angle = float(self.json_actions[action]['steering_angle']) * math.pi / 180.0
        throttle = float(self.json_actions[action]['speed'])

        continous_action = [steering_angle, throttle]

        return super().step(continous_action)


G_evaluation_env = DeepRacerDiscreteEnv()


import numpy as np
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):

            #t1 = time.time()

            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            #t2 = time.time()

            #print('Runner:',(t2-t1)*1000)

            #print('class Runner, run, self.obs[:]  ', self.obs[0,19200,0])

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch ofa steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)



# obs, returns, masks, actions, values, neglogpacs, states = runner.run()

#thisis called on 6 different values:
#eg: mb_obs have shape: 50, 1, 120, 160, 1 ==> 50, 120, 160, 1
# mb_returns have shape: 50, 1 ===> (50,) it is 50, empty
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape


    return_value = arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

    return return_value

# Network architecture
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch

Global_shape_batch = 1

def nature_cnn(unscaled_images, time_input, sampling_input, **conv_kwargs):
    global Global_shape_batch
    """
    CNN from Nature paper.
    """

    #session = tf.Session()

    #original shape is: [ 50 120 160   1]
    #print(session.run(tf.shape(unscaled_images)))

    #get one the value from unscaled_images and see if we can print it

    #i = tf.placeholder(tf.int32, shape=[None,1])
    #y = tf.slice(unscaled_images, i, [None, 1])



    #trying to reshape the observation
    #unscaled_images_shaped = tf.reshape(unscaled_images, [-1, 120, 160, 1])
    #print(session.run(tf.shape(unscaled_images_shaped)))



    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    #scaled_images = tf.cast(unscaled_images_shaped, tf.float32) / 255.

    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    h4 = activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


    h5 = tf.concat(axis=1, values=[time_input, sampling_input, h4])

    h6 = activ(fc(h5, 'fc2', nh=64, init_scale=np.sqrt(2)))


    #placeholder for the time and its concatenation
    #concat([t1, t2], 0)
    #time_input = tf.placeholder(dtype=tf.float32, shape=(Global_shape_batch,1), name='time_layer')

    #h6 = tf.concat(axis=1, values=[time_input, sampling_input, h5])

    #print('nature_cnn: shape of h5: ', session.run(tf.shape(h5)))
    return h6


def cnn(**conv_kwargs):
    def network_fn(X, time_input, sampling_input):
        return nature_cnn(X, time_input, sampling_input, **conv_kwargs)
    return network_fn



import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.models import get_network_builder

import gym


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, time_input, sampling_input, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.time_input = time_input#tf.placeholder(dtype=tf.float32, shape=(Global_shape_batch,1), name='time_layer')
        self.X = observations

        self.sampling_input = sampling_input#sampling input


        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        self.action2 = self.pd.mode()


        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess

        latency = observation[:,119,159,0].reshape(-1,1).astype(float)

        sampling = observation[:,119,158,0].reshape(-1,1).astype(float)


        #display_observation(observation[0,:,:,0])

        #print('latency is:',latency,sampling)

        feed_dict = {self.X: adjust_shape(self.X, observation)}

        #feeding the time input here
        feed_dict[self.time_input] = latency

        #feeding the sampling input here
        feed_dict[self.sampling_input] = sampling


        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        #print('_evaluate feed_dict:',feed_dict)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        #print('In step function:',observation.shape) #[-1, 120, 160, 1]


        #observation = observation.reshape((-1, 120, 160, 1))
        #print('In step function:',observation.shape)

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp


    def step2(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        #print('In step function:',observation.shape) #[-1, 120, 160, 1]


        #observation = observation.reshape((-1, 120, 160, 1))
        #print('In step function:',observation.shape)

        a, v, state, neglogp = self._evaluate([self.action2, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


# This the changed function to add extra variables

def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    global Global_shape_batch
    if isinstance(policy_network, str):
        #network_type = policy_network

        print('policy_kwargs:',policy_kwargs)
        #print('estimate_q:',estimate_q)

        policy_network = cnn(**policy_kwargs)



    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        global Global_shape_batch

        ob_space = spaces.Box(low=0, high=255,
                                            shape=(120,160, 1), dtype=np.uint8)

        Global_shape_batch = nbatch

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        time_input = tf.placeholder(dtype=tf.float32, shape=(nbatch,1), name='time_layer')

        sampling_input = tf.placeholder(dtype=tf.float32, shape=(nbatch,1), name='sampling_layer')

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x, time_input, sampling_input)


        _v_net = value_network


        #value network is also the copy of the policy network
        _v_net = policy_network
        vf_latent = _v_net(encoded_x, time_input, sampling_input)


        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            time_input = time_input,
            sampling_input = sampling_input,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms


import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize


MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling

            print('calling for act_model:', nbatch_act)
            act_model = policy(nbatch_act, 1, sess)

            print('calling for train_model:',nbatch_train)

            # Train model for training
            train_model = policy(nbatch_train, nsteps, sess)



        #print('microbatch_size is:',microbatch_size)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer

        self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]


        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step

        self.step2 = act_model.step2

        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        #print('shape of obs:',obs.shape)
        #obs = obs.reshape((-1,120,160,1))

        latency = obs[:,119,159,0].reshape(-1,1).astype(float)

        sampling = obs[:,119,158,0].reshape(-1,1).astype(float)

        #obs = obs[:,:19200,].reshape(-1, 120,160,1)

        #print('latency is:',latency, sampling)


         #feeding the time input here
        time_input = latency

        sampling_input = sampling

        td_map = {
            self.train_model.X : obs,
            self.train_model.time_input : time_input,
            self.train_model.sampling_input : sampling_input,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]


latencies =       [10,      10,     20,     20,   40,     40,    60,   60,    80,    80,    100,   100,  120,  120]
sampling_sleeps = [0.033, 0.033, 0.033, 0.033, 0.040, 0.040, 0.060, 0.060, 0.080, 0.080, 0.100, 0.100, 0.120, 0.120]
directions = [1, 2, 1, 2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
steps = 5000


def run_environment2(env, model, latency, sampling_sleep, direction):

    env.sampling_sleep = sampling_sleep
    env.sampling_rate = 1.0/(env.sampling_sleep)
    env.latency = latency

    #we are reversing direction on every episode
    env.reverse_dir = not env.reverse_dir

    obs = env.reset()
    rewards = 0
    done = False

    steps = 0

    while not done:

        env.sampling_sleep = sampling_sleep
        env.sampling_rate = 1.0/(env.sampling_sleep)
        env.latency = latency

        obs = obs.reshape(1,120,160,1)

        actions, _, _, _= model.step2(obs)
        obs, rew, done, _ = env.step(actions[0])

        rewards = rewards + rew

        steps = steps +1

        if done:
            obs = env.reset()

    return rewards, steps



def evaluate_model(model):
    print('Doing Evaluation ************')

    env = G_evaluation_env

    Total_reward = []
    Total_steps = []

    for epi in range(G_num_episodes_evaluation):
        for i in range(14):
            rew, steps = run_environment2(env, model, latencies[i], sampling_sleeps[i], directions[i])
            Total_reward.append(rew)
            Total_steps.append(steps)

            #print(epi, i, latencies[i], sampling_sleeps[i], directions[i], rew, steps)

    Total_reward = np.array(Total_reward)
    Total_steps = np.array(Total_steps)

    env.reverse_dir = False
    env.latency_steps = 0
    env.latency_index = 0
    env.latency = 10
    env.sampling_rate = 30.0
    env.sampling_sleep = (1.0/env.sampling_rate)

    return Total_reward.mean(), Total_steps.mean()



import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
#from baselines.common.policies import build_policy
#from baselines.ppo2.model import Model

MPI = None

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network, env, total_timesteps, eval_env = None, seed=0, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=3, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=3, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, **network_kwargs):

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    #print('network_kwargs:', network_kwargs, network)
    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = 1#env.num_envs

    # Get state_space and action_space
    #ob_space = env.observation_space
    #ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    model_fn = Model

    model = model_fn(policy=policy, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    if load_path is not None:
        model.load(load_path)


    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        #print("in def learn, obs is:", obs.shape)
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)


        mean_evaluation_reward, mean_evaluation_steps = evaluate_model(model)



        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)

            logger.logkv("mean_evaluation_reward", mean_evaluation_reward)
            logger.logkv("mean_evaluation_steps", mean_evaluation_steps)


            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)

    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

# environment building functions

from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import retro_wrappers
from baselines.common.wrappers import ClipActionsWrapper
from baselines.bench import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank =  0
    seed = seed
    logger_dir = logger.get_dir()


    def make_thunk(rank, initializer=None):
        print('make_thunk called:',rank)
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer
        )

    set_global_seeds(seed)

    return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])

def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}


    env  = DeepRacerDiscreteEnv()

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    return env

def build_env(args):
    ncpu = 1
    nenv = 1

    alg = args['alg']
    seed = args['seed']

    env_type, env_id = 'atari', args['env']
    env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args['gamestate'], reward_scale=args['reward_scale'])

    return env



def train(args, extra_args):
    env_type, env_id = 'atari', args['env']

    total_timesteps = int(args['num_timesteps'])
    seed = None

    alg_kwargs = atari()


    alg_kwargs.update(extra_args)

    env = build_env(args)

    alg_kwargs['network'] = 'cnn'

    print('Training {} on {}:{} with arguments \n{}'.format(args['alg'], env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


extra_args = {}
args = {'alg':'ppo2', 'env':'RoboMaker-DeepRacer-v101', 'env_type':None, 'gamestate':None,
        'log_path':None, 'network':None, 'num_env':None, 'num_timesteps':100000000,
        'play':False, 'reward_scale':1.0, 'save_path':None, 'save_video_interval':0,
        'save_video_length':200, 'seed':None}


rank = 0
logger.configure(log_path)

def atari():
    return dict(
        nsteps=nsteps_size, nminibatches=nminibatches_size,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.001,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.2,
        value_network='copy'
    )



model, env = train(args, extra_args)
