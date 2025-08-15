#!/usr/bin/env python3
"""
Example RL Training Script for Tessellate

This demonstrates how to use the Tessellate environment for RL research.
Includes a simple DQN implementation as a starting point.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tessellate_env import TessellateEnv
from evaluate_agents import evaluate_model_vs_random

class SimpleQNetwork(nn.Module):
    """Simple Q-network for Tessellate"""
    def __init__(self, input_size=101, hidden_size=256, output_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action

class DQNAgent:
    """Simple DQN agent for Tessellate"""
    
    def __init__(self, learning_rate=0.001, epsilon=0.1, gamma=0.95):
        self.q_network = SimpleQNetwork()
        self.target_network = SimpleQNetwork()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.epsilon = epsilon
        self.gamma = gamma
        self.update_target_every = 100
        self.steps = 0
    
    def select_action(self, state, valid_actions, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).squeeze()
            
            # Mask invalid actions
            masked_q = torch.full((100,), -float('inf'))
            masked_q[valid_actions] = q_values[valid_actions]
            
            return masked_q.argmax().item()
    
    def remember(self, state, action, reward, next_state, done, next_valid):
        """Store transition in replay buffer"""
        self.memory.append((state, action, reward, next_state, done, next_valid))
    
    def replay(self, batch_size=32):
        """Train on batch from replay buffer"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states)
            
            # Mask invalid actions for each next state
            for i, (_, _, _, _, _, next_valid) in enumerate(batch):
                mask = torch.full((100,), -float('inf'))
                mask[next_valid] = 0
                next_q[i] += mask
            
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Decay exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

def train_dqn(n_episodes=1000, eval_every=100):
    """Train a DQN agent on Tessellate"""
    
    env = TessellateEnv(reward_mode='mixed')
    agent = DQNAgent(epsilon=0.3)
    
    episode_rewards = []
    win_rates = []
    
    print("Starting DQN Training on Tessellate...")
    print("-" * 60)
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=True)
            
            next_state, reward, done, info = env.step(action)
            next_valid = env.get_valid_actions() if not done else []
            
            agent.remember(state, action, reward, next_state, done, next_valid)
            
            if len(agent.memory) > 32:
                agent.replay()
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        # Periodic evaluation
        if (episode + 1) % eval_every == 0:
            # Quick evaluation against random
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg reward (last 100): {np.mean(episode_rewards[-100:]):.3f}")
            
            # You could add proper evaluation here
            # win_rate = evaluate_against_random(agent, n_games=20)
            # print(f"  Win rate vs random: {win_rate:.1f}%")
    
    return agent, episode_rewards

def self_play_training(n_iterations=10, games_per_iteration=100):
    """
    Self-play training loop
    Train agent against itself, periodically updating opponent
    """
    env = TessellateEnv(reward_mode='mixed')
    agent = DQNAgent(epsilon=0.2)
    opponent = DQNAgent(epsilon=0.1)  # Less exploration for opponent
    
    print("Starting Self-Play Training...")
    print("-" * 60)
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        
        # Play games against current opponent
        for game in range(games_per_iteration):
            state = env.reset()
            done = False
            player_1_turn = True
            
            while not done:
                valid_actions = env.get_valid_actions()
                
                if player_1_turn:
                    action = agent.select_action(state, valid_actions, training=True)
                else:
                    action = opponent.select_action(state, valid_actions, training=False)
                
                next_state, reward, done, info = env.step(action)
                
                if player_1_turn:
                    next_valid = env.get_valid_actions() if not done else []
                    agent.remember(state, action, reward, next_state, done, next_valid)
                    
                    if len(agent.memory) > 32:
                        agent.replay()
                
                state = next_state
                player_1_turn = not player_1_turn
            
            if (game + 1) % 20 == 0:
                print(f"  Game {game + 1}/{games_per_iteration}")
        
        # Update opponent to current agent
        opponent.q_network.load_state_dict(agent.q_network.state_dict())
        agent.decay_epsilon(decay_rate=0.9)  # Bigger decay between iterations
        
        print(f"  Updated opponent, epsilon now: {agent.epsilon:.3f}")
    
    return agent

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--self-play":
        print("=== Self-Play Training Demo ===\n")
        agent = self_play_training(n_iterations=5, games_per_iteration=50)
    else:
        print("=== DQN Training Demo ===\n")
        agent, rewards = train_dqn(n_episodes=200, eval_every=50)
        
        print("\n=== Training Complete ===")
        print(f"Final avg reward: {np.mean(rewards[-50:]):.3f}")
    
    # Save the trained model
    torch.save(agent.q_network.state_dict(), "dqn_tessellate.pt")
    print("\nModel saved to dqn_tessellate.pt")