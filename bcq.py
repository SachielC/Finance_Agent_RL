import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym import spaces
import yfinance as yf
import pandas as pd

# ------------------------
# Enhanced Stock Trading Environment
# ------------------------
class StockTradingEnv(gym.Env):
    def __init__(self, tickers, start_date, end_date, initial_cash=10000):
        super(StockTradingEnv, self).__init__()
        self.tickers = tickers
        self.data = {}
        for ticker in tickers:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date)
            df = df.reset_index()
            df['Date'] = df['Date'].apply(lambda x: x.toordinal())
            df = df[['Date', 'Close', 'Dividends', 'Volume']]
            self.data[ticker] = df.reset_index(drop=True)
        
        self.max_steps = min([len(df) for df in self.data.values()]) - 1
        self.current_step = 0
        self.initial_cash = initial_cash
        self.positions = {ticker: 0.0 for ticker in self.tickers}

        self.observation_dim = len(self.tickers) * 4
        self.action_dim = len(self.tickers)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)

        # Track portfolio value for reward calculation
        self.last_value = initial_cash
        self.transaction_cost = 0.001  # 0.1% transaction fee

    def _get_state(self):
        features = []
        for ticker in self.tickers:
            df = self.data[ticker]
            row = df.iloc[self.current_step]
            features.extend([row['Close'], row['Dividends'], row['Volume'], row['Date']])
        return np.array(features, dtype=np.float32)

    def step(self, action):
        current_prices = {ticker: self.data[ticker].iloc[self.current_step]['Close'] 
                        for ticker in self.tickers}
        current_value = sum(self.positions[ticker] * current_prices[ticker] 
                        for ticker in self.tickers)

        # Convert action to weights with temperature
        weights = F.softmax(torch.tensor(action) * (1.0 + 0.1 * np.random.randn()))  # Add noise
        weights = weights.numpy()
        weights /= weights.sum()

        # Calculate target values with transaction cost penalty
        target_values = weights * current_value
        transaction_penalty = 0
        new_positions = {}
        
        for i, ticker in enumerate(self.tickers):
            new_shares = target_values[i] / current_prices[ticker]
            shares_diff = abs(new_shares - self.positions[ticker])
            transaction_penalty += shares_diff * current_prices[ticker] * self.transaction_cost
            new_positions[ticker] = new_shares

        self.positions = new_positions
        self.current_step += 1

        # Calculate new value with next prices
        next_prices = {ticker: self.data[ticker].iloc[self.current_step]['Close'] 
                     for ticker in self.tickers}
        new_value = sum(self.positions[ticker] * price for ticker, price in next_prices.items())
        new_value -= transaction_penalty

        # Reward components
        value_change = new_value - self.last_value
        momentum = sum((next_prices[t] - current_prices[t])/current_prices[t] 
                     for t in self.tickers) / len(self.tickers)
        
        reward = value_change * 10 + momentum * 0.5  # Scaled combined reward
        self.last_value = new_value

        done = self.current_step >= self.max_steps
        next_state = self._get_state()
        
        return next_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.last_value = self.initial_cash
        self.positions = {ticker: 0.0 for ticker in self.tickers}
        return self._get_state()

# ------------------------
# Replay Buffer (Unchanged)
# ------------------------
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind])
        )

# ------------------------
# BCQ Components with Improvements
# ------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 512)
        self.l5 = nn.Linear(512, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)
        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
        latent_dim = action_dim * 2
        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device
        self.grad_clip = 1.0

    def select_action(self, state, exploration_noise=0.2):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            state_repeat = state.repeat(100, 1)
            action_samples = self.vae.decode(state_repeat)
            perturbed_actions = self.actor(state_repeat, action_samples)
            q1 = self.critic.q1(state_repeat, perturbed_actions)
            ind = q1.argmax(0)
        action = perturbed_actions[ind].cpu().data.numpy().flatten()
        # Add decaying exploration noise
        return np.clip(action + exploration_noise * np.random.randn(self.action_dim), -1, 1)

    def train(self, replay_buffer, iterations, batch_size=256):
        for _ in range(iterations):
            # Sample batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # VAE training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip)
            self.vae_optimizer.step()

            # Critic training
            with torch.no_grad():
                next_state_rep = next_state.repeat_interleave(10, 0)
                next_action = self.actor_target(next_state_rep, self.vae.decode(next_state_rep))
                target_Q1, target_Q2 = self.critic_target(next_state_rep, next_action)
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].unsqueeze(1)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()

            # Actor training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Update targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ------------------------
# Enhanced Training Loop
# ------------------------
def train_agent(env, agent, replay_buffer, episodes=100, 
               batch_size=256, start_timesteps=2000,
               exploration_decay=0.995):
    total_timesteps = 0
    exploration_noise = 0.5
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        exploration_noise *= exploration_decay
        
        for t in range(env.max_steps):
            total_timesteps += 1
            
            if total_timesteps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, exploration_noise)
            
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward

            if total_timesteps >= start_timesteps:
                agent.train(replay_buffer, iterations=5, batch_size=batch_size)

            if done:
                break

        print(f"Episode {episode+1} | Reward: {episode_reward:.2f} | Noise: {exploration_noise:.3f}")

# ------------------------
# Main Execution
# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    
    env = StockTradingEnv(tickers, start_date, end_date)
    state_dim = env.observation_dim
    action_dim = env.action_dim
    max_action = 1.0
    
    agent = BCQ(state_dim, action_dim, max_action, device)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    train_agent(
        env, agent, replay_buffer,
        episodes=200,
        batch_size=512,
        start_timesteps=2000,
        exploration_decay=0.995
    )