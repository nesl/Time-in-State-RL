## Summary
This repo contains the code to train deep reinforcement learning policies for HalfCheetah, Ant, and the 1/18th scale autonomous car (DeepRacer). 
**[Slides](https://www.youtube.com/watch?v=rUTs2T_3A5Q&feature=emb_title)**,    **[Paper](https://corlconf.github.io/corl2020/paper_219/)**

```
Paper: Sim2Real Transfer for Deep Reinforcement Learning with Stochastic State Transition Delays, CoRL-2020.
```

</br>

```
The deployment heterogeneities and runtime compute stochasticity results 
in variable timing characteristics of sensor sampling rates and end-to-end delays from sensing to actuation. 
Prior works have used the technique of domain randomization to enable the successful transfer of policies 
across domains having different state transition delays. We show that variation in sampling rates and 
policy execution time leads to degradation in Deep RL policy performance, and that domain randomization 
is insufficient to overcome this limitation. We propose the Time-in-State RL (TS-RL) approach, which includes 
delays and sampling rate as additional agent observations at training time to improve the robustness of Deep RL 
policies. We demonstrate the efficacy of TSRL on HalfCheetah, Ant, and car robot in 
simulation and on a real robot using a 1/18th scale car.
```

</br>

## Domain randomization (DR) vs Time-in-State (TS) 

### Autonomous car at 100 ms latency
<table>
  <tr>
    <td> <img src="Short_Dr_video.gif" width="450"/></td>
    <td> <img src="Short_TS_video.gif" width="450"/></td>
   </tr> 
   <tr>
    <td>DR</td>
    <td>TS</td>
   </tr> 
</table>

### HalfCheetah at 20.6 ms latency
<table>
  <tr>
    <td> <img src="Ch_DR_5x.gif" width="450"/></td>
    <td> <img src="Ch_TS_5x.gif" width="450"/></td>
   </tr> 
  <tr>
    <td>DR</td>
    <td>TS</td>
   </tr> 
</table>

### Ant at 20.6 ms latency
<table>
  <tr>
    <td> <img src="Ant_DR_5x.gif" width="450"/></td>
    <td> <img src="Ant_TS_5x.gif" width="450"/></td>
   </tr> 
  <tr>
    <td>DR</td>
    <td>TS</td>
   </tr> 
</table>

</br>
</br>
</br>

## Demo video
**Check out the quick demo** of the transfer of policies from simulation to a real car robot.

[![TSRL Demo Video](demo_pic.png)](https://www.youtube.com/watch?v=5PlOerNRA9k)


</br>

## Code credits

a) The fully connected policy training using PPO code is taken from [open AI baselines](https://github.com/openai/baselines) with modifications done to allow
variable timing characteristics during training by fusing the delay observations with the neural network for images/state.

b) The code to train recurrent policies using PPO with variable timing characteristics is modified from the batch PPO code available [from Google-Research](https://github.com/google-research/batch-ppo).

c) The HalfCheetah environment and robot are taken from the [Pybullet](https://github.com/bulletphysics/bullet3) code with modifications for the variable timing characteristics for simulation steps.

d) The Ant environment and robot are taken from the [Pybullet](https://github.com/bulletphysics/bullet3) code with modifications for the variable timing characteristics for simulation steps.

e) Deepracer simulator modified by taking a snapshot from the open-source code of deepracer available [here](https://github.com/aws-robotics/aws-robomaker-sample-application-deepracer).
The changes include the track color, the captured camera sampling rate, and the removal of the AWS dependencies. The deepracer environment is modified to allow the variable timing characteristics.


## Requirements:
**1. Install the following requirements to train all the policies mentioned in the paper:**

a) [OpenAI gym](https://github.com/openai/gym), [OpenAI baselines](https://github.com/openai/baselines), [batch-PPO](https://github.com/google-research/batch-ppo)

b) [Gazebo and Ros](http://gazebosim.org/) for deepracer robotic car policies.

c) [Pybullet](https://github.com/bulletphysics/bullet3) for Ant and HalfCheetah


**2. To train only the fully policies only for Ant and Halfcheetah:**

a) [OpenAI gym](https://github.com/openai/gym), [OpenAI baselines](https://github.com/openai/baselines)

b) [Pybullet](https://github.com/bulletphysics/bullet3) for Ant and HalfCheetah

**3. To train only the recurrent policies for Halfcheetah:**

a) [OpenAI gym](https://github.com/openai/gym), [batch-PPO](https://github.com/google-research/batch-ppo)

b) [Pybullet](https://github.com/bulletphysics/bullet3) for HalfCheetah

**4. To train only the policies for Deepracer robotic car:**

a) [OpenAI gym](https://github.com/openai/gym), [OpenAI baselines](https://github.com/openai/baselines)

b) [Gazebo and Ros](http://gazebosim.org/) for deepracer robotic car policies.


## Usage
The training of policies, benchmarking, and visualization for each task is explained in the respective folders: 'deepracer', 'ant', 'halfcheetah', and 'halfcheetah-recurrent'.

## Questions
For any help/issue in running the code, please reachout to *sandha.iitr@gmail.com*
