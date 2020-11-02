# Code credits: The model loading code is taken from open AI baselines with modifications done to allow
# variable timing characteristics during evaluation. The deepracer environment is taken from the aws deepracer github
# code with modifications for the variable timing characteristics.


# Changing the sampling and latency input to the model

#multiple models and multiple paths can be added
path1 = 'Deepracer-checkpoints/ts_117.pb'  #Path of saved model

#path2 = 'ts_tf_frozen_model.pb'  #Path of saved model
#path3 = 'ts_tf_frozen_model.pb'  #Path of saved model

#paths = [path1, path2, path3]
paths = [path1]

#the folder to save the data
data_folder = 'data_ts/'

#the data saved in in folder with this name
#experiments = ['ts_1', 'ts_2', 'ts_3']
experiments = ['ts_1']

latencies = [20, 20, 40, 40, 60, 60, 80, 80, 100, 100, 120, 120]
sampling_sleeps = [0.033, 0.033, 0.040, 0.040, 0.060, 0.060, 0.080, 0.080, 0.100, 0.100, 0.120, 0.120]
directions = [ 1, 2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

#number of continuous steps to run the car
steps = 5000



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

            #image = do_randomization(image)

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


#         if self.latency_steps == self.latency_max_num_steps:

#             #update the latency
#             self.latency_index = (self.latency_index+1) % (len(self.latencies))
#             self.latency = self.latencies[self.latency_index]


#             #update the sampling rate
#             self.sampling_rate_index  = random.randint(0,1)
#             self.sampling_rate = self.sampling_rates[self.sampling_rate_index]

#             self.sampling_sleep = (1.0/self.sampling_rate)

#             if (self.latency/1000.0)> self.sampling_sleep: # match sampling input to the model and latency
#                  self.sampling_rate = 1000.0/self.latency


#             self.latency_steps = 0


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




import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True


import json

with open('custom_files/model_metadata.json', 'r') as f:
    model_metadata = json.load(f)
    json_actions = model_metadata['action_space']



def get_session(frozen_graph):
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())


    with tf.Graph().as_default() as graph:
        tf.import_graph_def(restored_graph_def, name="",input_map=None)

    x = graph.get_tensor_by_name('ppo2_model/Ob:0')
    y = graph.get_tensor_by_name('ppo2_model/pi_1/add:0')

    input_latency = graph.get_tensor_by_name('ppo2_model/time_layer:0')
    input_sampling = graph.get_tensor_by_name('ppo2_model/sampling_layer:0')


    sess = tf.Session(graph=graph, config=config)

    return sess, x,y, input_latency, input_sampling


# Automate the testing of the models
#Runs a simple setting

def test_in_simulator(sess, latency, sampling_sleep, total_steps, direction, x, y, input_latency, input_sampling):

    time_taken = []
    Actions_Taken = [] #stores speed and steering of the actions taken
    total_rewards = []

    env = DeepRacerDiscreteEnv()
    env.sampling_sleep = sampling_sleep
    env.sampling_rate = 1.0/(env.sampling_sleep)
    #print(env.sampling_sleep, env.sampling_rate)
    env.latency = latency
    env.dist_and_speed = []
    if direction==2: # when 2, we want to reverse the direction
        env.reverse_dir = not env.reverse_dir

    latency_input = np.ones((1,1))
    latency_input = latency_input*env.latency


    sampling_input = np.ones((1,1))
    sampling_input = sampling_input*env.sampling_rate


    steps_done = 0
    local_steps = 0
    obs = env.reset()

    #warmup
    obs = obs.reshape(1,120,160,1)
    action = sess.run(y, feed_dict={x: obs, input_latency:latency_input, input_sampling:sampling_input})




    while local_steps<=total_steps:
        done = False
        obs = env.reset()
        while not done and local_steps<=total_steps:
            t1 = time.time()
            obs = obs.reshape(1,120,160,1)

            action = sess.run(y, feed_dict={x: obs, input_latency:latency_input, input_sampling:sampling_input})
            action = np.argmax(action)


            steering_angle = json_actions[action]['steering_angle']
            throttle = json_actions[action]['speed']

            Actions_Taken.append([steering_angle,throttle])

            #updating the exact model runtime
            env.model_running_time = (time.time() - t1)

            obs, rew, done, _ = env.step(action)

            total_rewards.append(rew)

            t2 = time.time()

            time_taken.append(t2-t1)
            local_steps = local_steps + 1

            if done:
                obs = env.reset()
                print('Local_steps:',local_steps)

    dist_and_speed = env.dist_and_speed

    del env

    return total_rewards, Actions_Taken, dist_and_speed, time_taken


# save the data
import pickle

def save_data(path, total_rewards, dist_and_speed, Actions_Taken):
    with open(path, 'wb') as f:
        print("Saving the data", path)
        data = [total_rewards, dist_and_speed, Actions_Taken]
        pickle.dump(data, f)


def do_testing(sess, x, y, input_latency, input_sampling,  exp_name):
    for i in range(len(latencies)):
        latency = latencies[i]
        sampling_sleep = sampling_sleeps[i]
        direction = directions[i]

        total_rewards, Actions_Taken, dist_and_speed, time_taken = test_in_simulator(sess, latency, sampling_sleep, steps, direction,  x, y, input_latency, input_sampling)

        path = data_folder +  exp_name+'_'+str(direction)+'_'+str(latency)

        save_data(path, total_rewards, dist_and_speed, Actions_Taken)

        del total_rewards, Actions_Taken, dist_and_speed, time_taken


for i in range(len(paths)):
    frozen_graph = paths[i]
    exp_name = experiments[i]

    sess, x,y, input_latency, input_sampling = get_session(frozen_graph)

    do_testing(sess,x,y, input_latency, input_sampling, exp_name)

    del sess, x, y, input_latency, input_sampling
