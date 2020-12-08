import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
import collections as cc
import random as rn
from matplotlib import pylab as plt

device = T.device("cuda" if T.cuda.is_available() else "cpu")

GAMMA: float = 0.99
EXPLORE: int = 20000
EPOCH: int = 100000
MAX_MOVES: int = 2500
INITIAL_EPSILON: float = 0.1
FINAL_EPSILON: float = 0.0001
REPLAY_MEMORY: int = 1000000
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.0001
ANTAGONISTIC_MOVE_RATE: float = 0.02
STEPS: int = 256


def dis(_size: int):
    """
    Generate a numpy array with a discrete distribution using softmax

    :param _size:integer size of distribution
    :return:  numpy array
    """
    _ = np.random.rand(_size)
    _d = np.exp(_) / sum(np.exp(_))
    return _d


class Grid(object):

    def __init__(self, state_size: int):
        """
        Init the game

        self.state_size: int: integer the size of the numpy array
        self.state_prob: np.array (linear): a probability distribution form 0 to 1 each index is a symbolic
                                    representation of a geographically area. The higher probability would represent
                                    a population density. For example the more people the greater chance of an
                                    ambulance is required
        self.antagonistic_state: np.array (linear): numpy zero filled array that represented the current state
        self.sympathetic_state: np.array (linear): numpy zero filled array for the current location of an ambulance
        self.reward: float: init the reward value

        self.sympathetic_state[0]: set the ambulance to index[0]

        :param state_size: the size of the numpy array
        """
        self.state_size: int = state_size
        self.antagonistic_state: np.array = dis(self.state_size)
        self.sympathetic_state: np.array = np.zeros(self.state_size)
        self.sympathetic_state[0] = 1.0
        self.reward: float = 0.0
        self.reward_count = []

    def antagonistic_move(self, antagonistic_move_rate: float) -> object:
        """
        The idea of the antagonistic_move is to change the state at a given interval based on a given probability
                distribution.
        :param antagonistic_move_rate: integer
        """

        if rn.random() < antagonistic_move_rate:
            self.antagonistic_state = dis(len(self.antagonistic_state))

    def sympathetic_move(self, action: int) -> object:
        """
        Move the agent to the left or right by increasing or decreasing the agents current numpy index value.
                The action check to see the move will be outside the array.

        :param action: 0 to move right and 1 to move left
        """

        if action == 0:
            _idx = self.sympathetic_state.argmax()
            if _idx < (self.sympathetic_state.size - 1):
                self.sympathetic_state[_idx] = 0
                self.sympathetic_state[_idx + 1] = 1
        elif action == 1:
            _idx = self.sympathetic_state.argmax()
            if _idx > 0:
                self.sympathetic_state[_idx] = 0
                self.sympathetic_state[_idx - 1] = 1

    def sympathetic_reward(self) -> object:
        """
        The reward method rewards or punishes the agent if index vale is not equal the index value of the patient.

            argmax(antagonistic_state) -> [0.02138027 0.01243725 0.32124701 ... ]
            argmax(sympathetic_state) ->  [0             0           1      ...]

        :return: reward for the agent of the current state
        """
        if np.argmax(self.antagonistic_state) == np.argmax(self.sympathetic_state):
            self.reward += 50
        else:
            self.reward -= 1

        self.reward_count.append(self.reward)


class LNN(nn.Module):
    """

    """

    def __init__(self, l1: int, l2: int, l3: int, output_size: int):
        super(LNN, self).__init__()
        self.linear1 = nn.Linear(l1, l2)
        self.linear2 = nn.Linear(l2, l3)
        self.linear3 = nn.Linear(l3, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


"""
Main function

:var network_e is the experience network
:var network_t is the target network
"""

network_e = LNN(128, 256, 128, 2).to(device)
network_t = LNN(128, 256, 128, 2).to(device)
network_t.load_state_dict(network_e.state_dict())

optimizer = op.Adam(network_e.parameters(), lr=LEARNING_RATE)

loss_value = []
epsilon = INITIAL_EPSILON
replay_memory = cc.deque(maxlen=REPLAY_MEMORY)
move_count = 0
target = 0
# learn_steps = 0
target_sync = 50
reward_count = np.array([])

for _e in range(EPOCH):

    game = Grid(128)
    state_t = T.from_numpy(game.antagonistic_state).float().to(device)

    for steps in range(STEPS):
        move_count += 1
        target += 1

        _rn = rn.random()
        Q = network_e(state_t)
        if _rn < epsilon:
            action_index = rn.randint(0, 1)
        else:
            Q_ = Q.detach().cpu().numpy()
            action_index = np.argmax(Q_)

        game.sympathetic_move(action_index)
        game.sympathetic_reward()
        reward_ = game.reward
        reward_count = np.append(reward_count, reward_)

        done = 1 if reward_ > 0 else 0

        state = game.antagonistic_state

        game.antagonistic_move(ANTAGONISTIC_MOVE_RATE)
        state_ = game.antagonistic_state

        experience = (state, action_index, reward_, state_, done)
        replay_memory.append(experience)

        state = state_
        # learn_steps += 1

        if len(replay_memory) > BATCH_SIZE:
            batch_ = rn.sample(replay_memory, BATCH_SIZE)

            batch_state = T.tensor([s1 for (s1, a, r, s2, d) in batch_]).float().to(device)
            batch_action = T.tensor([a for (s1, a, r, s2, d) in batch_]).float().to(device)
            batch_reward = T.tensor([r for (s1, a, r, s2, d) in batch_]).float().to(device)
            batch_state_ = T.tensor([s2 for (s1, a, r, s2, d) in batch_]).float().to(device)
            batch_done = T.tensor([d for (s1, a, r, s2, d) in batch_]).to(device)

            batch_1 = network_e(batch_state)

            with T.no_grad():
                batch_2 = network_t(batch_state_)

            Y = batch_reward + GAMMA * ((1 - batch_done) * T.max(batch_2, dim=1)[0])
            X = batch_1.gather(dim=1, index=batch_action.long().unsqueeze(dim=1)).squeeze()

            loss = F.mse_loss(X, Y.detach())

            optimizer.zero_grad()
            loss.backward()

            loss_value.append(loss.item())
            optimizer.step()

            if target % target_sync == 0:
                network_t.load_state_dict(network_e.state_dict())

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        # print(loss)
    if move_count > MAX_MOVES:
        break

    # if _e % 10 == 0:
    #   T.save(network_e.state_dict(), 'ddqn-policy.para')

loss_value = np.array(loss_value)
plt.figure(figsize=(10, 7))
plt.plot(loss_value)
plt.xlabel("iterations", fontsize=22)
plt.ylabel("Loss", fontsize=22)

print(game.reward_count)
plt.figure(figsize=(10, 7))
plt.plot(reward_count)
plt.xlabel("iterations", fontsize=22)
plt.ylabel("Count", fontsize=22)
