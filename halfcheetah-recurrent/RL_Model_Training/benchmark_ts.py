# Benchmark Time-in-State recurrent policies


# Written by Sandeep Singh Sandha, UCLA, sandha.iitr@gmail.com
# using various function from Batch-ppo original code


logdir = 'trained_checkpoints/'

runs = ['20201025T074614-cheetah_ts/']
checkpoints = ['model.ckpt-117007000']
save_names = ['TS1']

num_of_episodes = 10

latency_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sampling_index = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]




import tensorflow as tf
import os

current_directory = os.getcwd()
logdir = os.path.join(current_directory, logdir)

from agents import algorithms
from agents.scripts import networks

##Half-Cheetah Environment from PyBullet

myseed = 0
G_TS = True

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

import random
import numpy as np
#import tensorflow as tf


np.random.seed(myseed)
random.seed(myseed)

import gym
import numpy as np

import sys
print(sys.executable)

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


#from scene_stadium import SinglePlayerStadiumScene

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data

from pybullet_envs.scene_abstract import Scene
import pybullet


class StadiumScene(Scene):
  zero_at_running_strip_start_line = True  # if False, center of coordinates (0,0,0) will be at the middle of the stadium
  stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
  stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
  stadiumLoaded = 0

  def episode_restart(self, bullet_client):
    self._p = bullet_client
    Scene.episode_restart(self, bullet_client)  # contains cpp_world.clean_everything()
    if (self.stadiumLoaded == 0):
      self.stadiumLoaded = 1

      # stadium_pose = cpp_household.Pose()
      # if self.zero_at_running_strip_start_line:
      #	 stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants

      filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
      self.ground_plane_mjcf = self._p.loadSDF(filename)
      #filename = os.path.join(pybullet_data.getDataPath(),"stadium_no_collision.sdf")
      #self.ground_plane_mjcf = self._p.loadSDF(filename)
      #
      for i in self.ground_plane_mjcf:
        self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
        self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,i)

      #	for j in range(p.getNumJoints(i)):
      #		self._p.changeDynamics(i,j,lateralFriction=0)
      #despite the name (stadium_no_collision), it DID have collision, so don't add duplicate ground


class SinglePlayerStadiumScene(StadiumScene):
  "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
  multiplayer = False


class MultiplayerStadiumScene(StadiumScene):
  multiplayer = True
  players_count = 3

  def actor_introduce(self, robot):
    StadiumScene.actor_introduce(self, robot)
    i = robot.player_n - 1  # 0 1 2 => -1 0 +1
    robot.move_robot(0, i, 0)


from robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
import numpy as np
import pybullet
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
        #print(state.shape)

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



#from scene_stadium import SinglePlayerStadiumScene
from env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet
#from robot_locomotors import HalfCheetah

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

    #used to evolve the latency
    self.prev_action = None

    self.original_timestep = (0.0165*1000.0)/4.0


    #used to enable jitter
    self.episodic_l = 0.0
    self.episodic_si = G_sampling_min

    self.rewards_list = []
    self.total_epi_reward = 0

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

#         print(self.latency, self.sampling_interval)

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True


    if self.steps == G_T_Horizon:
        done = True
        self.rewards_list.append(self.total_epi_reward)
        self.total_epi_reward = 0.0

    #update the state with the timing information
    if G_TS:
        state[26] = self.latency/self.latency_max
        state[27] = self.sampling_interval/self.latency_max

    #print('Rewards:', self.rewards)
    #return state, sum(self.rewards), bool(done), {}
    self.total_epi_reward= self.total_epi_reward + reward
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


import os

import gym
#import tensorflow as tf

from agents import tools
from agents.scripts import utility


def _define_loop(graph, eval_steps):
  """Create and configure an evaluation loop.

  Args:
    graph: Object providing graph elements via attributes.
    eval_steps: Number of evaluation steps per epoch.

  Returns:
    Loop object.
  """
  loop = tools.Loop(
      None, graph.step, graph.should_log, graph.do_report, graph.force_reset)
  loop.add_phase(
      'eval', graph.done, graph.score, graph.summary, eval_steps,
      report_every=eval_steps,
      log_every=None,
      checkpoint_every=None,
      feed={graph.is_training: False})
  return loop

def _create_environment(config,sampling_index, latency_index):
  """Constructor for an instance of the environment.

  Args:
    config: Object providing configurations via attributes.
    outdir: Directory to store videos in.

  Raises:
    NotImplementedError: For action spaces other than Box and Discrete.

  Returns:
    Wrapped OpenAI Gym environment.
  """
#   if isinstance(config.env, str):
#     env = gym.make(config.env)
#   else:
#     env = config.env()

  env = HalfCheetahBulletEnv()
  
  #env.sampling_interval= sampling_interval
  #env.latency = latency
  env.latency = latency_index*G_lat_inc
  env.sampling_interval = sampling_index*env.sampling_interval_min
  #print("Setting : ",env.latency,env.sampling_interval)

  # Ensure that the environment has the specification attribute set as expected
  # by the monitor wrapper.
  if not hasattr(env, 'spec'):
    setattr(env, 'spec', getattr(env, 'spec', None))
  if config.max_length:
    env = tools.wrappers.LimitDuration(env, config.max_length)
  #env = gym.wrappers.Monitor(env, outdir, lambda unused_episode_number: True, force=True, video_callable=False)
  #env = gym.wrappers.Monitor(env, outdir, force=True, video_callable=False)
  #env = gym.wrappers.Monitor(env, outdir, force=True)

  if isinstance(env.action_space, gym.spaces.Box):
    env = tools.wrappers.RangeNormalize(env)
    env = tools.wrappers.ClipAction(env)
  elif isinstance(env.action_space, gym.spaces.Discrete):
    env = tools.wrappers.RangeNormalize(env, action=False)
  else:
    message = "Unsupported action space '{}'".format(type(env.actions_space))
    raise NotImplementedError(message)
  env = tools.wrappers.ConvertTo32Bit(env)
  env = tools.wrappers.CacheSpaces(env)
  return env

def run_setting(config, checkpoint, sampling_index,latency_index,num_episodes):

    with tf.device('/cpu:0'):
        batch_env = utility.define_batch_env(lambda: _create_environment(config, sampling_index, latency_index),num_agents=1, env_processes=False)
        #batch_env.sampling_interval= sampling_interval
        #batch_env.latency = latency

        graph = utility.define_simulation_graph(
            batch_env, config.algorithm, config)
        total_steps = num_episodes * config.max_length
        loop = _define_loop(graph, total_steps)

    saver = utility.define_saver(exclude=(r'.*_temporary.*', r'global_step'))
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        utility.initialize_variables(
            sess, saver, config.logdir, checkpoint, resume=True)
        for unused_score in loop.run(sess, saver, total_steps):
          pass

    tf.reset_default_graph()

    return batch_env.rewards_list


import pickle


for i in range(len(checkpoints)):
    model = logdir+ runs[i] + checkpoints[i]
    #print('*'*100)
    #print(logdir + runs[i])
    config = utility.load_config(logdir + runs[i])

    for j in range(len(latency_index)):
        sampling_j= sampling_index[j]
        latency_j = latency_index[j]
        res = run_setting(config, model, sampling_j, latency_j, num_of_episodes)
        print(i, j,res)

        file_name = 'data/'+save_names[i]+'_'+str(sampling_j)+'_'+str(latency_j)+'.p'
        pickle.dump([res, sampling_j, latency_j, model, config], open(file_name, "wb" ) )
