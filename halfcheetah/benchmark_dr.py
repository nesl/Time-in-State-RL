 # Code credits: The model loading code is taken from open AI baselines with modifications done to allow
 # variable timing characteristics during evaluation. The Cheetah environment and robot is taken from the Pybullet
 # code with modifications for the variable timing characteristics.

# Add below the path of the trained checkpoints. A sample is shown below
# Multiple paths can be added  to the list below for more evaluations
path1 = 'hc_dr_policies/checkpoints/hc_dr_saved1'

#path2 and path3 are not used. The code can be modified to use them
#by changing the lines 17 and line 26.
path2 = 'checkpoint_save_path/checkpoints/01595'
path3 = 'checkpoint_save_path/checkpoints/01595'

paths = [path1]

#modify to use all the three checkpoints
#paths = [path1, path2, path3]

#folder to save the benchmarking data
save_folder = 'data_dr'

#Names used to saved the data for each checkpoint
saved_models = ['dr1']

#modify to use all the three checkpoints
#saved_models = ['dr1', 'dr2', 'dr3']


tick = (0.0165*1000.0/4.0)


latencies =  [tick*0, tick*1, tick*2, tick*3, tick*4, tick*5, tick*6, tick*7, tick*8, tick*9, tick*10]
samplings =  [tick*1, tick*1, tick*2, tick*3, tick*4, tick*5, tick*6, tick*7, tick*8, tick*9, tick*10]

G_TS = False

myseed = 0

G_T_Horizon = 1000
G_T_Steps = 10000

G_delay_max = (0.0165*1000.0/4.0)*10.0
G_sampling_min = (0.0165*1000.0/4.0)*1.0


G_max_num_steps = G_T_Horizon

G_Tick = (0.0165*1000.0/4.0)

G_Action_repeated = True

G_policy_selection_sample = True

G_Action_clip = 1.0

G_evaluation_env = None #note this need to set correctly
G_use_checkpoint_evaluaion = True
G_evaluate_every = 1
G_evaluation_inc = (0.0165*1000.0/4.0)
G_num_episodes_evaluation = 1


G_lat_inc = (0.0165*1000.0/4.0)#G_delay_max/(G_T_Steps/G_max_num_steps)

G_lat_inc_steps = 10.0#G_delay_max/G_lat_inc


G_enable_latency_jitter = False

#jitter of one tick-rate: no used in code. Jitter of 1 tick_rate is hard coded in the code
G_latency_jitter = 1
G_sampling_jitter = 1


import tensorflow as tf
import random
import numpy as np

tf.set_random_seed(myseed)
np.random.seed(myseed)
random.seed(myseed)

from scene_stadium import SinglePlayerStadiumScene

from robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
import numpy as np
import pybullet
import os
import pybullet_data
from robot_bases import BodyPart


class WalkerBase(MJCFBasedRobot):

  def __init__(self, fn, robot_name, action_dim, obs_dim, power):
    MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
    self.power = power
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz = [0, 0, 0]

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    for j in self.ordered_joints:
      j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

    self.feet = [self.parts[f] for f in self.foot_list]
    self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  #this is the function where action torque is applied to the joints
  def apply_action(self, a):
    assert (np.isfinite(a).all())

    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -G_Action_clip, +G_Action_clip)))

  #IMP: This function gets the next state from the robot
  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_real_xyz = body_pose.xyz()
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
        [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)

    timing_info_holder = np.array([0.0, 0.0], dtype=np.float32)



    if G_TS:
        #state = np.clip(np.concatenate([more] + [j] + [self.feet_contact], [timing_info_holder]), -5, +5)
        state = np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

        state = np.concatenate((state, timing_info_holder))

    else:
        state = np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)


    return state

    #return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

  def calc_potential(self):
    # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    # all rewards have rew/frame units and close to 1.0
    debugmode = 0
    if (debugmode):
      print("calc_potential: self.walk_target_dist")
      print(self.walk_target_dist)
      print("self.scene.dt")
      print(self.scene.dt)
      print("self.scene.frame_skip")
      print(self.scene.frame_skip)
      print("self.scene.timestep")
      print(self.scene.timestep)
    return -self.walk_target_dist / self.scene.dt
class HalfCheetah(WalkerBase):
  foot_list = ["ffoot", "fshin", "fthigh", "bfoot", "bshin",
               "bthigh"]  # track these contacts with ground

  def __init__(self):
        if G_TS:
            WalkerBase.__init__(self, "half_cheetah.xml", "torso", action_dim=6, obs_dim=28, power=0.90)


        else:
            WalkerBase.__init__(self, "half_cheetah.xml", "torso", action_dim=6, obs_dim=26, power=0.90)


  def alive_bonus(self, z, pitch):
    # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
    return +1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[
        2] and not self.feet_contact[4] and not self.feet_contact[5] else -1

  def robot_specific_reset(self, bullet_client):
    WalkerBase.robot_specific_reset(self, bullet_client)
    self.jdict["bthigh"].power_coef = 120.0
    self.jdict["bshin"].power_coef = 90.0
    self.jdict["bfoot"].power_coef = 60.0
    self.jdict["fthigh"].power_coef = 140.0
    self.jdict["fshin"].power_coef = 60.0
    self.jdict["ffoot"].power_coef = 30.0



from env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet

class WalkerBaseBulletEnv(MJCFBaseBulletEnv):

  def __init__(self, robot, render=False):
    # print("WalkerBase::__init__ start")
    self.camera_x = 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.stateId = -1
    MJCFBaseBulletEnv.__init__(self, robot, render)


    self.time_tick = G_Tick  #1ms

    self.latency = 0.0 # save the latency of most recent returned state
    self.latency_max = G_delay_max # max latency in ms


    self.max_num_steps = G_max_num_steps # for steps latency will be fixed or change on reset or done after G_max_num_steps.
    self.latency_steps = 0
    self.steps = 0


    self.sampling_interval = G_sampling_min
    self.sampling_interval_min = G_sampling_min #30 Hz frequency

    #increase the latency within thresholds
    self.index = 1


    #used to evolve the latency
    self.prev_action = None

    self.original_timestep = (0.0165*1000.0)/4.0


    #used to enable jitter
    self.episodic_l = 0.0
    self.episodic_si = G_sampling_min


  #This is the place where simulation parameters are configured that are applied in the step.
  #Scene definitions are: https://github.com/bulletphysics/bullet3/blob/aae8048722f2596f7e2bdd52d2a1dcb52a218f2b/examples/pybullet/gym/pybullet_envs/scene_stadium.py
  # - https://github.com/bulletphysics/bullet3/blob/aec9968e281faca7bc56bc05ccaf0ef29d82d062/examples/pybullet/gym/pybullet_envs/scene_abstract.py

  def create_single_player_scene(self, bullet_client):
#     self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
#                                                   gravity=9.8,
#                                                   timestep=0.0165 / 4,
#                                                   frame_skip=4)

    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=9.8,
                                                  timestep=(self.time_tick/1000.0),
                                                  frame_skip=4)

    return self.stadium_scene


  def reset(self):
    if (self.stateId >= 0):
      #print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)

    r = MJCFBaseBulletEnv.reset(self)

    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
        self._p, self.stadium_scene.ground_plane_mjcf)
    self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                            self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("saving state self.stateId:",self.stateId)


    self.prev_action =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.steps = 0

    #update the state with the timing information
    if G_TS:
        r[26] = self.latency/self.latency_max
        r[27] = self.sampling_interval/self.latency_max


    return r

  def _isDone(self):
    return self._alive < 0

  def move_robot(self, init_x, init_y, init_z):
    "Used by multiplayer stadium to move sideways, to another running lane."


    self.cpp_robot.query_position()
    pose = self.cpp_robot.root_part.pose()
    pose.move_xyz(
        init_x, init_y, init_z
    )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
    self.cpp_robot.set_pose(pose)

  electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
  stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
  foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  joints_at_limit_cost = -0.1  # discourage stuck joints



  #given an action calculate the reward based on the robot current state
  def calreward(self,a):
    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch

    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0

    electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean())
    # let's assume we have DC motor with controller, and reverse current braking

    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

    rewards = [
        self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
    ]
    self.HUD(state, a, done)
    rewards= sum(rewards)



    return rewards


  def step(self, a):

#     if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
#       self.robot.apply_action(a)
#       self.scene.global_step()


    self.latency_steps = self.latency_steps + 1
    self.steps = self.steps + 1

    latency = (self.latency)

    reward = 0

    local_sim_steps = 0

    if G_Action_repeated:
        #simulate the latency
        if latency>0:
            for i in range(int(latency/self.time_tick)):
                self.robot.apply_action(self.prev_action)
                self.scene.global_step()

                reward = reward + self.calreward(a)

                local_sim_steps = local_sim_steps + 1


        #print('local_sim_steps:', local_sim_steps)



        #simulate the sampling interval
        if self.sampling_interval>self.latency:
            delay = (self.sampling_interval - self.latency)
            for i in range(int(delay/self.time_tick)):
                self.robot.apply_action(a)
                self.scene.global_step()

                reward = reward + self.calreward(a)

                local_sim_steps = local_sim_steps + 1


    else:
        #simulate the latency
        if latency>0:
            self.robot.apply_action(self.prev_action)
            for i in range(int(latency/self.time_tick)):
                self.scene.global_step()

                reward = reward + self.calreward(a)

                local_sim_steps = local_sim_steps + 1

        #simulate the sampling interval
        if self.sampling_interval>self.latency:
            delay = (self.sampling_interval - self.latency)
            self.robot.apply_action(a)
            for i in range(int(delay/self.time_tick)):
                self.scene.global_step()

                reward = reward + self.calreward(a)

                local_sim_steps = local_sim_steps + 1



    if local_sim_steps>0:
        reward = reward/local_sim_steps # we are rescaling the reward based on local_sim_steps

    #print('local_sim_steps:', local_sim_steps)

    self.prev_action = a

    #update the latency and sampling as needed
#     if self.latency_steps == self.max_num_steps:

#         #print(self.latency, self.sampling_interval)

#         self.latency = self.index*G_lat_inc

#         self.sampling_interval = self.sampling_interval_min

#         if self.latency>self.sampling_interval:
#             self.sampling_interval = self.latency


#         self.episodic_l = self.latency  #used to maintain jitter for an episode
#         self.episodic_si = self.sampling_interval ##used to maintain jitter for an episode


#         self.latency_steps = 0

#         if self.index==int(G_lat_inc_steps):
#             self.index = -1

#         self.index = self.index + 1


    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

#     potential_old = self.potential
#     self.potential = self.robot.calc_potential()
#     progress = float(self.potential - potential_old)

#     feet_collision_cost = 0.0
#     for i, f in enumerate(
#         self.robot.feet
#     ):  # TODO: Maybe calculating feet contacts could be done within the robot code
#       contact_ids = set((x[2], x[4]) for x in f.contact_list())
#       #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
#       if (self.ground_ids & contact_ids):
#         #see Issue 63: https://github.com/openai/roboschool/issues/63
#         #feet_collision_cost += self.foot_collision_cost
#         self.robot.feet_contact[i] = 1.0
#       else:
#         self.robot.feet_contact[i] = 0.0

#     electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean())
#     # let's assume we have DC motor with controller, and reverse current braking


#     electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

#     joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

#     debugmode = 0

#     if (debugmode):
#       print("alive=")
#       print(self._alive)
#       print("progress")
#       print(progress)
#       print("electricity_cost")
#       print(electricity_cost)
#       print("joints_at_limit_cost")
#       print(joints_at_limit_cost)
#       print("feet_collision_cost")
#       print(feet_collision_cost)

#     self.rewards = [
#         self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
#     ]
#     if (debugmode):
#       print("rewards=")
#       print(self.rewards)
#       print("sum rewards")
#       print(sum(self.rewards))

#     self.HUD(state, a, done)
#     self.reward += sum(self.rewards)



    if self.steps == G_T_Horizon:
        done = True


    if G_enable_latency_jitter:
            #add jitter in latency# 5 ms jitter
            jitter = random.randint(-1,1)

            self.latency = self.episodic_l + jitter*G_lat_inc

            if self.latency<0:
                self.latency = 0.0

            jitter = random.randint(-1,1)

            self.sampling_interval = self.episodic_si + jitter*G_lat_inc


            if self.latency>self.sampling_interval:
                self.sampling_interval = self.latency

            if self.sampling_interval < self.sampling_interval_min:
                self.sampling_interval = self.sampling_interval_min


    #update the state with the timing information
    if G_TS:
        state[26] = self.latency/self.latency_max
        state[27] = self.sampling_interval/self.latency_max

    #print('Rewards:', self.rewards)
    #return state, sum(self.rewards), bool(done), {}
    return state, reward, bool(done), {}

  def camera_adjust(self):
    x, y, z = self.robot.body_real_xyz

    self.camera_x = x
    self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)

class HalfCheetahBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render=False):
    self.robot = HalfCheetah()
    WalkerBaseBulletEnv.__init__(self, self.robot, render)

  def _isDone(self):
    return False



from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module


# Network architectures for the Pendulum

# Network architecture
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch

def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn


# Policy maintenance functions which create the network architecture

import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
#from baselines.common.models import get_network_builder

import gym


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, sess=None, **tensors):
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

        self.X = observations
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
        #self.action = self.pd.sample() #self.pd.mode()

        self.action = self.pd.mode()

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
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

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

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
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

def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network

        policy_network = mlp(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
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
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

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
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
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
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
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


import os
import time
import numpy as np
import os.path as osp
from baselines import logger


MPI = None

def constfn(val):
    def f(_):
        return val
    return f

def get_saved_model(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=1, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, **network_kwargs):

    total_timesteps = int(total_timesteps)

    #print('network_kwargs:', network_kwargs, network)
    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = 1#env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch
    #is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    return model


def atari():
    return dict(
        nsteps=0, nminibatches=64,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.001,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.2,
        value_network='copy'
    )


args = {'alg':'ppo2', 'env':'', 'env_type':None, 'gamestate':None,
        'log_path':None, 'network':None, 'num_env':None, 'num_timesteps':1000000.0,
        'play':False, 'reward_scale':1.0, 'save_path':None, 'save_video_interval':0,
        'save_video_length':200, 'seed':0}

env_type, env_id = 'atari', args['env']

total_timesteps = int(args['num_timesteps'])
seed = None


env = HalfCheetahBulletEnv()
alg_kwargs = atari()
alg_kwargs['network'] = 'mlp'


model = get_saved_model(
        env=env,
        seed=None,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )


def run_environment(model, latency, sampling):
    env = HalfCheetahBulletEnv()

    env.latency = latency
    env.sampling_interval = sampling

    obs = env.reset()
    rewards = 0
    done = False

    #for i in range(500):
    while not done:
        actions, _, _, _= model.step(obs)
        obs, rew, done, _ = env.step(actions[0])

        rewards = rewards + rew

        if done:
            obs = env.reset()

    return rewards


import pickle

num_episodes = 10

total_data = []

model_name = 0
for path in paths:

    model.load(path)

    model_data = []

    for i in range(len(latencies)):
        latency = latencies[i]
        sampling = samplings[i]

        run_data = []

        for j in range(num_episodes):
            reward = run_environment(model, latency, sampling)

            run_data.append(reward)


        #save the run_data for the model
        saved_path = save_folder+'/' + saved_models[model_name]+ '_'+ str(i)

        #print('saved_path', saved_path)

        try:
            current_directory = os.getcwd()
            final_directory = os.path.join(current_directory, save_folder)
            if not os.path.exists(final_directory):
                os.makedirs(final_directory)
        except:
            continue

        with open(saved_path, 'wb') as f:
            print("Saving the data", saved_path)
            data = [run_data, path, latency,sampling]
            pickle.dump(data, f)


        run_data = np.array(run_data)

        print(run_data.mean())
        model_data.append(run_data.mean())

    model_name +=1

    total_data.append(model_data)
