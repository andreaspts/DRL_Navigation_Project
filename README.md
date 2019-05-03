# DRL_Navigation_Project
Navigation project: Train an agent via a deep reinforcement algorithm to complete a task in a large 3d world

For an introduction to reinforcement learning and deep reinforcement learning in particular as a applied to a similar training problem, we refer to [Cartpole - Introduction to Reinforcement Learning (DQN - Deep Q-Learning)](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288).

### Introduction

<p align="center">
  <img width="460" height="300" src="banana world.gif">
</p>

In this project we train an agent to explore a world and to collect yellow and blue bananas. More precisely, the agent receives a reward of +1 for collecting a yellow banana and is punished for collecting a blue banana. Hence, the overall aim of the agent is to amass as many yellow bananas as possible while refraining from collecting blue ones. The problem is defined as an episodic task and the environment is considered to be solved when an anverage score of +13 over 100 consecutive episodes is attained.

The environment is further specified by a continuous state space of 37 dimensions and by a discrete action space. The latter consists of the actions corresponding to moving forward, backward, left and right. For completeness, we give the Unity details of the environment:

```
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```
### Getting Started

### Instructions
