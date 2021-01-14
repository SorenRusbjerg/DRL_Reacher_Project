# Initial agent
In this project a DDPG agent is implemented. The agent is based on the Udacity DDPG-pendulum agent, and initial parameter set used for training based on the paper https://arxiv.org/pdf/1509.02971.pdf. 

# Final implementation

## Actor-Critic network
Actor-critic network is implemented in the file `model.py` 
* First Layer is a batch normalization layer to normalize input signals variance
* Both Actor and Critic networks have 2 hidden Linear layers of 256 and 164 neurons with Relu activation 
* Output layer of Critic has a tanh-activation function to make the output range [-1,1]
* Actor does not use activation on the output layer
* Gradient clipping is used in the critic network to prevent exploding gradients.

## Agent parameters
The agent is implemented in `ddpg_agent.py`.

Below the parameters used for training the final model:
```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.2e-4       # learning rate of the actor 
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
STEPS_PER_UPDATE = 10   # Nr of steps before agent update
UPDATES_PER_BATCH = 2   # Nr of learning updates per new batch
NOISE_DECAY = 0.997     # Decay per episode of OU noise variance
ACTOR_FC1 = 256         # 1 actor layer neurons
ACTOR_FC2 = 164         # 2 actor layer neurons
CRITIC_FC1 = 256        # 1 critic layer neurons
CRITIC_FC2 = 164        # 2 critic layer neurons
```

The DDPG uses noise from a Ornstein-Uhlenbeck process added to the action outputs to add exploration to the agent. The gaussion input noise to this process is decresed over episodes until a minimum level of 10% initial value.

The agent parameters are updated from a `Replay buffer` 2 times every 10 samples using batches of 64 samples.

## Agent description
An actor network is used to estimate action values together with a critic network that uses the action values is input together with the state values to estimate the Q-function (action-value function).  
Actor and critic network is divided into a target and a local network. The local networks parameters are trained using gradient steps with their respective loss functions. The target networks are used to estimate the next Q-value when training the critic network. 
The target networks are slowly updated towards the local networks using a soft-update function to get the target network parameters. 

All the agent state-action-reward-state are saved into a replay buffer during training  and sampled randomly in the agent update step.

In the end the agent got an average score of 37.8 over 100 episodes. 

## Improvement steps done during project 
Below is a list of updates and steps made gradually to produce the final agent implementation:
* Include Batch normalization layers
* Changing random seed for network
* only updating agent after N steps 
* Test with high process noise
* Test with low process noise
* Include Gradient clipping
* Implement Reducing exploration, by gradually decreasing process noise
* Changes random OU noise from uniform to gausian distribution
* Test with bigger Network size
* Test with smaller Network size
* Test with reduced Learning rates

In the end the reduced learning rate together with a smaller network size seemed to make the agent train and gets average scores above 1. 
Also the decreasing OU process noise seems to help here. 
In the folder `/Training examples` different agent version training steps can be found as `*.html` files. 
The final version is found [here](./Training%20examples/Navigation_ddqn_1800episodes_BN_medium_net_Low_LR2_NoiseLowerLimIncreased_2.html)

Final training score plot:
![image info](./Training%20examples/Training.PNG)


## Future improvements
* Update agent to handle multi agent environment.
* Make priority replay buffer





