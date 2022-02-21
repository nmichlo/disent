import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ========================================================================= #
# BEGIN                                                                     #
# ========================================================================= #


class Net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 120, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0, std=1)
        self.fc2 = nn.Linear(120, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)


class GridWorld():

    def __init__(self, grid_size, start_pos, goal_pos, reward_value):
        self.player_state = start_pos
        self.goal = goal_pos
        self.reward = reward_value
        self.player_start = start_pos
        self.env_dims = grid_size
        self.game_state = np.zeros(grid_size)

    def simulate_action(self, action):
        sim_state = np.zeros(self.env_dims)
        if not self.player_state[0] == 0 and action == 0:
            sim_state[self.player_state + (-1, 0)] = 1
        if not self.player_state[1] == (self.env_dims[1] - 1) and action == 1:
            sim_state[self.player_state + (0, 1)] = 1
        if not self.player_state[0] == (self.env_dims[0] - 1) and action == 2:
            sim_state[self.player_state + (1, 0)] = 1
        if not self.player_state[1] == 0 and action == 3:
            sim_state[self.player_state + (0, -1)] = 1
        return sim_state

    def take_action(self, action):
        self.game_state[self.player_state] = 0
        if not self.player_state[0] == 0 and action == 0:
            self.player_state = self.player_state + (-1, 0)
        if not self.player_state[1] == (self.env_dims[1] - 1) and action == 1:
            self.player_state = self.player_state + (0, 1)
        if not self.player_state[0] == (self.env_dims[0] - 1) and action == 2:
            self.player_state = self.player_state + (1, 0)
        if not self.player_state[1] == 0 and action == 3:
            self.player_state = self.player_state + (0, -1)
        self.game_state[self.player_state] = 1

    def return_rewards(self):
        if self.player_state == self.goal:
            return self.reward
        else:
            return -1

    def reset_player(self):
        self.player_state = self.player_start
        self.game_state[self.player_state] = 1


# ========================================================================= #
# RL                                                                       #
# ========================================================================= #


def run_rl(
    num_steps: int = 100,
    num_trajectories: int = 20,
    step_size: float = 0.1,
    discount: float = 0.1,
    input_dim: int = 20,
):

    losses = np.zeros(num_trajectories)

    domain = GridWorld((10, 10), (0, 0), (9, 9), 100)
    net = Net(input_dim)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=step_size, momentum=0.0)

    for traj in range(num_trajectories):
        domain.reset_player()

        for step in range(num_steps):
            optimizer.zero_grad()

            outputs = net(domain.game_state)
            domain.take_action(np.argmax(outputs))
            next_Qs = [net(domain.simulate_action(i)) for i in range(4)]
            loss = criterion(outputs, domain.return_rewards + discount * next_Qs)
            loss.backward()
            optimizer.step()

            # print statistics
            losses[traj] = losses[traj] + loss.item()

        print(losses[traj])

    print('Finished Training')


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == "__main__":
    run_rl()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


# EXMAPLE:
# encoder
# data
# loading
# representations

# torch.load()

# map back xy to dataset image
# -- example

# send json files with representations linked to xy positions
