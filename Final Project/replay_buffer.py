import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean

from ale_py import ALEInterface
from ale_py.roms import SpaceInvaders
from PIL import Image
import time
import random
import pandas as pd


#initialize the spaceinvaders ROM
ale = ALEInterface()
ale.loadROM(SpaceInvaders)

env = gym.make("ALE/SpaceInvaders-v5",obs_type='rgb',repeat_action_probability=0)#render_mode="human")
#env = gym.make("CartPole-v1")


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# CUDA from GPU or no
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ",device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#image processing to turn pic in grey scale and pic pixels of interest
def image_mod(image,start,end):
    
    smallx = 210
    bigx = 0
    
    smally = 210
    bigy = 0
    
    for i in range(180,195):
        for j in range(160):
            #print(image[i,j])
            #print(image[i,j])
            if image[i,j][1] == 132:#> 120 and image[i,j][1] < 160:
                image[i,j] = [255,0,0]
                if i < smallx:
                    smallx = i
                if i > bigx:
                    bigx = i
                if j > bigy:
                    bigy = j
                if j < smally:
                    smally = j
    #print(bigy,bigx,smally,smallx)
    for i in range(smallx,bigx):
        image[i,bigy] = [255,0,0]
    
    for i in range(smallx,bigx):
        image[i,smally] = [255,0,0]
    
    space_size = 40
    
    spaceL = int(space_size/2)
    spaceR = int(space_size/2)
    
    if bigx == 0 and bigy == 0 and smallx == 210 and smally == 210:
        pass
    else:
        
        overall = 28#55
        space_size = overall - (bigy - smally) 
        
        spaceL = int(space_size/2)
        spaceR = int(space_size/2)       
        while spaceL + spaceR + (bigy - smally) > overall:
            spaceL -= 1
        while spaceL + spaceR + (bigy - smally) < overall:
            spaceL += 1 
        if smally < spaceL:
            #print('here')
            spaceL = smally
            spaceR = spaceR*2 - spaceL
            
            while spaceL + spaceR + (bigy - smally) > overall:
                spaceR -= 1
            while spaceL + spaceR + (bigy - smally) < overall:
                spaceR += 1
        elif 210-bigy < spaceR:
            #print('there')
            spaceR = 210-bigy
            spaceL = spaceL*2 - 210-bigx
            while spaceL + spaceR + (bigy - smally) > overall:
                spaceL -= 1
            while spaceL + spaceR + (bigy - smally) < overall:
                spaceL += 1
        start = int(smally-spaceL)
        end = int(bigy+spaceR)
    output = rgb2gray(image[25:159,start:end]).flatten() #partial image
    #output = rgb2gray(image).flatten() ## full image
    return image, output, start, end
    

class ReplayMemory(object): #replay buffer for huber loss

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
class Optimization_ReplayMemory(object): #replay buffer for extra training

    def __init__(self,start,end,step):
        self.memory = {}
        self.lower_bounds = start
        self.higher_bounds = end
        for i in range(start,end,step):
            self.memory.update({i:[]})
        print(self.memory)
            
    def round_half_up(self,n, decimals=-2):
        multiplier = 10 ** decimals
        return math.floor(n*multiplier + 0.5) / multiplier

    def push(self, action_list,reward):
        
        action_lib = self.round_half_up(reward)
        
        
        #print("action_lib: ",action_lib)
        self.memory[action_lib].append(action_list)
        #print(self.memory)
    
    def replay(self):
        print("============================================================")
        found = []
        
        for i in range(int(self.higher_bounds)-100,int(self.lower_bounds)-100,-100):
            if len(found) != 0:
                break
            else:
                if len(self.memory[i]) != 0:
                    val = random.randint(0,len(self.memory[i])-1)
                    found = self.memory[i][val]
                    
        
        return found

    


class DQN(nn.Module): 
    #DQN with built in Neural network being used
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        val = 100#40
        #val = 100
        #val = 84 #space == 55
        #val2 = 612
        val2 = 129
        self.layer1 = nn.Linear(n_observations, val2*val)
        self.layer2 = nn.Linear(val2*val,val2*val)
        self.layer3 = nn.Linear(val2*val, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.0
EPS_DECAY = 100000
TAU = 0.05
LR = 1e-3

# Get number of actions from gym action space
n_actions = env.action_space.n
print("action count: ",n_actions)
# Get the number of state observations
state, info = env.reset()
start = 0
end = 28#55
image_in, state, start, end = image_mod(state,start,end)
n_observations = len(state)

print("len: ",n_observations)
print(state)

policy_net = DQN(n_observations, n_actions).to(device)#initiate policies
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
replay_buffer_storage = Optimization_ReplayMemory(0,2000,100) ############################################


steps_done = 0
episode_durations = []
highest_mean = 0


def select_action(state):#pick action (modify here to go with episolon decay or straight greedy)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if steps_done%500 == 0:
        print("threshold: ",eps_threshold)
    
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)




def plot_durations(show_result=False): #live ploting of the output data
    global highest_mean
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        print("mean: ",means[-1])
        if means[-1] > highest_mean:
            highest_mean = means[-1]
        print("highest mean: ",highest_mean)
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
if torch.cuda.is_available():
    num_episodes = 4000
else:
    num_episodes = 1
    print("only run once CPU operation too slow")

max_total_reward = 0

state_reward = []
average_reward = []
frame_count_list = []

for trial in range(0,4): #cycle of testing
    state_reward = []
    average_reward = []
    frame_count_list = []
    action_list = []
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    replay_buffer_storage = Optimization_ReplayMemory(0,2000,100) ############################################
    
    episode_count = 0
    for i_episode in range(100000): #hard set episode count
        episode_count += 1
        # Initialize the environment and get it's state
        state, info = env.reset()
        image_in, state, start, end = image_mod(state,start,end)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) #### update here state value
        total_reward = 0
        frame_count = 0
        
        total_action_list = []
        
        action_number = 0
        individual_reward_list = []
        individual_observation_list = []
        individual_terminated_list = []
        individual_truncated_list = []
        
        #random.randint(1,10000)
        
        if i_episode%10 == 0: #replay
            #if False:
            memory_instance = replay_buffer_storage.replay()
            if len(memory_instance) > 0:
                print(len(memory_instance[0]))
                print(len(memory_instance[1]))
                print(len(memory_instance[2]))
                print(len(memory_instance[3]))
                print(len(memory_instance[4]))
                
                for action_number in range(len(memory_instance[0])):
                    action = memory_instance[0][action_number]
                    observation = memory_instance[1][action_number]
                    reward = memory_instance[2][action_number]
                    terminated = memory_instance[3][action_number]
                    truncated = memory_instance[4][action_number]
                    total_reward += reward
                    image_in, observation, start, end = image_mod(observation,start,end)
                    reward = torch.tensor([reward], device=device)
                    done = terminated or truncated
                    
                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                        
                    state = next_state
                    optimize_model()
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    target_net.load_state_dict(target_net_state_dict)
            
                    if done:
                        episode_durations.append(total_reward)
                        plot_durations()
                        
                        #total_action_list = []
                        break
                tmp_reward = reward.cpu().numpy()[0]
                #frame_count_list.append(frame_count)
                #state_reward.append(tmp_reward)
                #average_reward.append(total_reward)
                print("final reward: ",total_reward,i_episode)
                print("max: ",max_total_reward)
                print("output: ",reward.cpu().numpy()[0])
                print()
                        
                    
            else:
                print("++++++++++++++++++++++++++")
        
        state, info = env.reset()
        image_in, state, start, end = image_mod(state,start,end)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) #### update here state value
        total_reward = 0
        frame_count = 0
        
        total_action_list = []
        
        action_number = 0
        individual_reward_list = []
        individual_observation_list = []
        individual_terminated_list = []
        individual_truncated_list = []
        
        
        if True: #exploration/exploitation
            frame = 0
            for t in count():
                frame += 1
                action = select_action(state)
                #print("action",action.cpu().numpy()[0][0])
                #total_action_list.append(action.cpu().numpy()[0][0])
                total_action_list.append(action)
                
                observation, reward, terminated, truncated, _ = env.step(action.item())
                im = Image.fromarray(observation)
                #if episode_count == 1:
                #    im.save("./random_vid/your_file"+str(frame)+".jpeg")
                
                #if episode_count == 2 or episode_count == 10 or episode_count == 20 or episode_count == 50 or episode_count == 100 or episode_count == 140 or episode_count == 180 or episode_count == 250:
                #    im.save("./random_vid_"+str(episode_count)+"/your_file"+str(frame)+".jpeg")    
                
                individual_reward_list.append(reward)
                individual_observation_list.append(observation)
                individual_truncated_list.append(truncated)
                individual_terminated_list.append(terminated)
                action_number += 1
                
                
                #print("step reward: ",reward)
                total_reward += reward
                if total_reward > max_total_reward:
                    max_total_reward = total_reward
                #print("total reward: ",total_reward)
                image_in, observation, start, end = image_mod(observation,start,end)
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
        
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
                # Store the transition in memory
                memory.push(state, action, next_state, reward)
        
                # Move to the next state
                state = next_state
        
                # Perform one step of the optimization (on the policy network)
                optimize_model()
        
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)
        
                if done:
                    episode_durations.append(total_reward)
                    plot_durations()
                    
                    #total_action_list = []
                    break
                frame_count += 1
            #print("action list: ",total_action_list)
            replay_buffer_storage.push([total_action_list,individual_observation_list,individual_reward_list,individual_terminated_list,individual_truncated_list],total_reward)
            tmp_reward = reward.cpu().numpy()[0]
            frame_count_list.append(frame_count)
            state_reward.append(tmp_reward)
            average_reward.append(total_reward)
            print("final reward: ",total_reward,i_episode)
            print("max: ",max_total_reward)
            print("output: ",reward.cpu().numpy()[0])
            print()
            #print(total_reward)
            #print([state_reward[:],average_reward[:]])
            if i_episode%1 == 0:
                print(state_reward)
                print(action_number)
                action_list.append(action_number)
                df = pd.DataFrame([state_reward,average_reward,action_list])
                df.to_csv('list'+str(trial)+'.csv', index=False)
            

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show(block=False)