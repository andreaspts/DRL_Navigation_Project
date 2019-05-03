# Project report

## Learning algorithm

The employed learning algorithm is the standard Deep Q-Learning algorithm which was introduced in the article [Human-level control through deep reinforcementlearning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). 

Due to the fact that we are using state vectors as an input and not image data we use a simple deep neural network instead of a convolutional neural network to determine the action-value function. The former consists of the following 5 layers coded into the model.py file:

- Fully connected layer - input: 37 (state size) output: 64
- Fully connected layer - input: 64 output: 32
- Fully connected layer - input: 32 output: 16
- Fully connected layer - input: 16 output: 8
- Fully connected layer - input: 8 output: 4 (action size)

We speficy the parameters used in the Deep Q-Learning algorithm (as in the dqn-function of the Navigation_solution.ipynb notebook):

- We set the number of episodes n_episodes to 3000. The number of episodes needed to solve the environment and reach a score of 13.0 is expected to be smaller.
- We set the maximum steps per episode max_t to 1000. 
- We start with an epsilion eps_start of 1.0.
- We end with an epsilion eps_end of 0.01.
- We set the epsilion decay rate eps_decay to 0.999.

Furthermore we give the parameters used in the dqn_agent.py file:

- The size of the replay buffer BUFFER_SIZE is set to 10^6.
- The mini batch size BATCH_SIZE is set to 64.
- The discount factor GAMMA for future rewards is set to 0.99.
- We set the value for soft update of target parameters TAU to 10^-3.
- The learning rate for the gradient descent LR is set to 5 * 10^-4.
- The update rate UPDATE_EVERY is set to 4 meaning that every 4 steps a gradient descent update at minibatch size is done.

## Results

With the above specifications we report the results.

First we give a plot of the scores over the episodes:

![results](images/plot.png)

Then we list the average score every 100 episodes up to the point where the agent reaches a score equal or higher than 13.0: 

```
Episode 100	Average Score: 0.16
Episode 200	Average Score: 0.56
Episode 300	Average Score: 1.67
Episode 400	Average Score: 2.33
Episode 500	Average Score: 3.13
Episode 600	Average Score: 4.27
Episode 700	Average Score: 5.53
Episode 800	Average Score: 6.19
Episode 900	Average Score: 6.95
Episode 1000	Average Score: 7.73
Episode 1100	Average Score: 7.24
Episode 1200	Average Score: 8.52
Episode 1300	Average Score: 8.92
Episode 1400	Average Score: 10.42
Episode 1500	Average Score: 10.52
Episode 1600	Average Score: 10.62
Episode 1700	Average Score: 11.35
Episode 1800	Average Score: 11.57
Episode 1900	Average Score: 11.44
Episode 2000	Average Score: 11.97
Episode 2100	Average Score: 12.41
Episode 2200	Average Score: 12.42
Episode 2300	Average Score: 12.40
Episode 2374	Average Score: 13.06
Environment solved in 2274 episodes!	Average Score: 13.06
```

## Possible future extensions of the setting

1. The hyperparameters should be optimized: For example, we could change the epsilon decay rate, the learning rate, the batch size and improve the network structure (more/less layers and units; overfitting could be tackled using dropout).

A bunch of improvements on the bare Deep Q-Learning have been suggested in the literature:

2. Double DQNs: This method is also known as double Learning and was introduced in the article [Double Q-learning](https://papers.nips.cc/paper/3964-double-q-learning). Using this method, overestimation of Q-values can be tackled.
3. Prioritized Experience Replay (PER): This method was introduced in the article [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952). This work is motivated by the idea to give priority to experiences which could be more important for completing the task.
4. Dueling Deep Q Networks (DDQNs): Dueling DQNs were introduced in the article [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581). The architecture proposed in this work allows to decompose Q(s,a) into a sum of the state-value function V(s) and the advantage A(s,a) which quantifies the improvement of taking a particular action compared to all other ones at the given state. By calculating V(s), the agent learns about the value of a state without having to learn about the impact of each available action at that state. This is useful when the effect of available actions of states onto the environment is not too important.
5. RAINBOW Paper
6. Learning from pixels
