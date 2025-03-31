import copy
import datetime
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
# Custom Stock Trading Environment
# ------------------------
class StockTradingEnv(gym.Env):
    """
    A simplified stock trading environment for multiple stocks.
    The state consists of the close price, dividend, volume, and date (as an ordinal number)
    for each stock. The action is a continuous vector (one per stock) in [-1,1] interpreted
    as a trading signal.
    """
    def __init__(self, tickers, start_date, end_date, initial_cash=1e6):
        super(StockTradingEnv, self).__init__()
        self.tickers = tickers
        self.data = {}
        for ticker in tickers:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date)
            # Make sure we have the columns we need. Rename "Dividends" to "Dividends" if needed.
            df = df.reset_index()
            # Convert date to ordinal (an increasing number)
            df['Date'] = df['Date'].apply(lambda x: x.toordinal())
            # Ensure we have 'Close', 'Dividends', and 'Volume'
            df = df[['Date', 'Close', 'Dividends', 'Volume']]
            self.data[ticker] = df.reset_index(drop=True)
        
        # For simplicity, assume all stocks have the same number of records
        self.max_steps = min([len(df) for df in self.data.values()])
        self.current_step = 0

        self.initial_cash = initial_cash
        self.cash = initial_cash
        # Positions: number of shares held per stock
        self.positions = {ticker: 0.0 for ticker in self.tickers}

        # Define state dimensions: for each stock we use 4 features.
        self.observation_dim = len(self.tickers) * 4
        self.action_dim = len(self.tickers)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        # We allow any real number in the state (features will be normalized later if needed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)

    def _get_state(self):
        """Aggregate features for all stocks at the current step into a flat vector."""
        features = []
        for ticker in self.tickers:
            df = self.data[ticker]
            row = df.iloc[self.current_step]
            # For each ticker we add: [close, dividend, volume, date]
            features.extend([row['Close'], row['Dividends'], row['Volume'], row['Date']])
        return np.array(features, dtype=np.float32)

    def step(self, action):
        """
        Execute trades based on the action.
        For each stock, interpret action[i] (in [-1,1]) as:
          - positive: buy, negative: sell
        We scale the trade size as a fraction of total portfolio value.
        """
        state = self._get_state()
        # Compute current portfolio value
        total_value = self.cash + sum(
            self.positions[ticker] * self.data[ticker].iloc[self.current_step]['Close'] 
            for ticker in self.tickers
        )
        # For each stock, compute the trade value
        for i, ticker in enumerate(self.tickers):
            current_price = self.data[ticker].iloc[self.current_step]['Close']
            # Here we use a scaling factor (e.g., 10% of portfolio per action)
            trade_value = action[i] * total_value * 0.1
            if trade_value > 0:  # buy
                # Buy shares if cash is available
                if self.cash >= trade_value:
                    shares_bought = trade_value / current_price
                    self.positions[ticker] += shares_bought
                    self.cash -= trade_value
            elif trade_value < 0:  # sell
                shares_to_sell = min(-trade_value / current_price, self.positions[ticker])
                self.positions[ticker] -= shares_to_sell
                self.cash += shares_to_sell * current_price

        prev_value = total_value

        self.current_step += 1
        done = (self.current_step >= self.max_steps - 1)
        # New portfolio value
        new_value = self.cash + sum(
            self.positions[ticker] * self.data[ticker].iloc[self.current_step]['Close'] 
            for ticker in self.tickers
        )
        # Reward is the change in portfolio value
        reward = new_value - prev_value
        next_state = self._get_state()
        return next_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = {ticker: 0.0 for ticker in self.tickers}
        return self._get_state()

# ------------------------
# Replay Buffer
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
# BCQ Agent Components (Actor, Critic, VAE)
# ------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
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
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

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

# Variational Auto-Encoder to generate actions close to those in the data
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
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, sample a latent vector if none provided
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

# ------------------------
# BCQ Agent
# ------------------------
class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device

    def select_action(self, state):
        # Select action by sampling multiple candidate actions and choosing the best one according to Q
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            state_repeat = state.repeat(100, 1)
            action_samples = self.vae.decode(state_repeat)
            perturbed_actions = self.actor(state_repeat, action_samples)
            q1 = self.critic.q1(state_repeat, perturbed_actions)
            ind = q1.argmax(0)
        return perturbed_actions[ind].cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):
        for it in range(iterations):
            # Sample replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # VAE Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training
            with torch.no_grad():
                # Duplicate next state 10 times for action sampling
                next_state_rep = next_state.repeat_interleave(10, 0)
                next_action = self.vae.decode(next_state_rep)
                next_action = self.actor_target(next_state_rep, next_action)
                target_Q1, target_Q2 = self.critic_target(next_state_rep, next_action)
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor Training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ------------------------
# Training Loop
# ------------------------
def train_agent(env, agent, replay_buffer, episodes=100, max_timesteps=200, batch_size=100, start_timesteps=1000):
    total_timesteps = 0
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            total_timesteps += 1
            # Use random action until the replay buffer is filled sufficiently
            if total_timesteps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward

            if total_timesteps >= start_timesteps:
                agent.train(replay_buffer, iterations=1, batch_size=batch_size)

            if done:
                break
        print(f"Episode {episode}: Reward {episode_reward:.2f}")

# ------------------------
# Main: Setup Environment, Agent, and Start Training
# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # For demonstration we use a few tickers. Replace with a full S&P500 list as needed.
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    env = StockTradingEnv(tickers, start_date, end_date)
    
    state_dim = env.observation_dim
    action_dim = env.action_dim
    max_action = 1.0  # Actions in [-1,1]
    
    agent = BCQ(state_dim, action_dim, max_action, device)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    # Train for a set number of episodes (adjust episodes and timesteps as needed)
    train_agent(env, agent, replay_buffer, episodes=50, max_timesteps=env.max_steps)
