import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import yfinance as yf

# --------------------------
# Trading Environment
# --------------------------
class TradingEnv(gym.Env):
    """
    Trading environment with daily rebalancing and debug logs.

    - Starts with $10,000 equally invested in 20 stocks.
    - Uses daily closing prices from yfinance.
    - At each step, the agent supplies a target allocation (vector that sums to 1).
      The environment rebalances by selling shares where the target is lower than
      current holdings and buying where it is higher.
    - The portfolio value is updated every day according to the current closing prices.
    - Reward is computed as the current portfolio value minus the starting cash ($10,000),
      representing the pure increase in portfolio value.
    - Debug logs print the day number, actions taken, portfolio value, and cash balance.
    """
    def __init__(self, stock_prices, initial_cash=10000):
        super().__init__()
        self.stock_prices = stock_prices  # shape: (T, num_stocks)
        self.initial_cash = initial_cash
        self.num_stocks = stock_prices.shape[1]
        self.current_step = 0

        # Action: allocation vector (continuous values) that should sum to 1.
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)
        # Observation: concatenation of current prices and current holdings.
        obs_low = np.zeros(self.num_stocks * 2, dtype=np.float32)
        obs_high = np.full(self.num_stocks * 2, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self):
        self.current_step = 0
        current_prices = self.stock_prices[self.current_step]

        # Distribute initial cash equally among stocks
        allocation = np.full(self.num_stocks, 1.0 / self.num_stocks)
        self.cash = self.initial_cash
        self.holdings = (allocation * self.cash) / current_prices
        self.cash = 0.0  # All cash is used to buy stocks

        self.cost_basis = current_prices.copy()
        self.sold_profit = 0.0
        return np.concatenate([current_prices, self.holdings])
    
    def _get_obs(self):
        current_prices = self.stock_prices[self.current_step]
        return np.concatenate([current_prices, self.holdings])
    def step(self, action):
        current_prices = self.stock_prices[self.current_step]
        portfolio_value = self.cash + np.sum(self.holdings * current_prices)

        # Normalize action so that it sums to 1
        allocation = np.array(action) / np.sum(action)
        target_value = portfolio_value * allocation
        target_shares = target_value / current_prices

        # Rebalance portfolio while ensuring no negative holdings
        for i in range(self.num_stocks):
            delta = target_shares[i] - self.holdings[i]
            if delta < 0:  # Sell shares
                shares_to_sell = min(-delta, self.holdings[i])  # Prevent selling more than owned
                sale_amount = shares_to_sell * current_prices[i]
                realized_profit = (current_prices[i] - self.cost_basis[i]) * shares_to_sell
                self.holdings[i] -= shares_to_sell
                self.cash += sale_amount
                self.sold_profit += realized_profit
            elif delta > 0:  # Buy shares if cash available
                cost = delta * current_prices[i]
                if cost > self.cash:
                    delta = self.cash / current_prices[i]
                    cost = delta * current_prices[i]
                if self.holdings[i] > 0:
                    self.cost_basis[i] = ((self.cost_basis[i] * self.holdings[i]) + (current_prices[i] * delta)) / (self.holdings[i] + delta)
                else:
                    self.cost_basis[i] = current_prices[i]
                self.holdings[i] += delta
                self.cash -= cost

        # Compute new portfolio value
        new_prices = self.stock_prices[self.current_step]
        portfolio_value = self.cash + np.sum(self.holdings * new_prices)

        # Reward is portfolio increase over the initial investment
        reward = portfolio_value - self.initial_cash

        self.current_step += 1
        done = self.current_step >= len(self.stock_prices)
        obs = self._get_obs() if not done else np.concatenate([self.stock_prices[-1], self.holdings])
        return obs, reward, done, {}
    def render(self, mode="human"):
        current_prices = self.stock_prices[self.current_step - 1]
        portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        print(f"Day {self.current_step} - Portfolio Value: ${portfolio_value:.2f}, Cash: ${self.cash:.2f}")

# --------------------------
# Replay Buffer
# --------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))
    def __len__(self):
        return len(self.buffer)

# --------------------------
# BCQ Networks
# --------------------------
# Q-Network: outputs a Q-value given state and action.
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# VAE: learns to reconstruct actions given state-action pairs.
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=10):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 256)
        self.e2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, latent_dim)
        self.log_std = nn.Linear(256, latent_dim)
        self.d1 = nn.Linear(state_dim + latent_dim, 256)
        self.d2 = nn.Linear(256, 256)
        self.d3 = nn.Linear(256, action_dim)
        self.latent_dim = latent_dim
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.e1(x))
        x = torch.relu(self.e2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        x = torch.cat([state, z], dim=1)
        x = torch.relu(self.d1(x))
        x = torch.relu(self.d2(x))
        action_recon = torch.tanh(self.d3(x))
        return action_recon, mean, std
    def decode(self, state, z=None):
        batch_size = state.size(0)
        if z is None:
            z = torch.randn(batch_size, self.latent_dim).to(state.device).clamp(-0.5, 0.5)
        x = torch.cat([state, z], dim=1)
        x = torch.relu(self.d1(x))
        x = torch.relu(self.d2(x))
        action = torch.tanh(self.d3(x))
        return action

# Perturbation network: adds a small adjustment to actions.
class Perturbation(nn.Module):
    def __init__(self, state_dim, action_dim, phi=0.05):
        super(Perturbation, self).__init__()
        self.phi = phi
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    def forward(self, state, action):
        delta = self.phi * self.net(torch.cat([state, action], dim=1))
        return delta

# --------------------------
# BCQ Agent
# --------------------------
class BCQAgent:
    def __init__(self, state_dim, action_dim, device, discount=0.99, tau=0.005, lmbda=0.75):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.q_target = QNetwork(state_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(self.q_network.parameters())

        self.vae = VAE(state_dim, action_dim).to(device)
        self.vae_optimizer = optim.Adam(self.vae.parameters())

        self.perturbation = Perturbation(state_dim, action_dim).to(device)
        self.perturbation_optimizer = optim.Adam(self.perturbation.parameters())
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            num_samples = 10
            state_repeat = state_tensor.repeat(num_samples, 1)
            action_samples = self.vae.decode(state_repeat)
            perturbed_actions = action_samples + self.perturbation(state_repeat, action_samples)
            q_values = self.q_network(state_repeat, perturbed_actions)
            best_index = q_values.argmax()
            best_action = perturbed_actions[best_index].cpu().numpy()
        # Map output from [-1,1] to [0,1] and normalize.
        best_action = (best_action + 1) / 2
        best_action = best_action / best_action.sum()
        return best_action

    def train(self, replay_buffer, batch_size=64):
        batch = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action = torch.FloatTensor(np.array(batch.action)).to(self.device)
        reward = torch.FloatTensor(np.array(batch.reward)).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done = torch.FloatTensor(np.array(batch.done)).to(self.device).unsqueeze(1)

        # VAE update.
        recon, mean, std = self.vae(state, action)
        recon_loss = nn.MSELoss()(recon, action)
        kl_loss = (-0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2, dim=1)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Q-network update.
        with torch.no_grad():
            num_samples = 10
            next_state_repeat = next_state.repeat(num_samples, 1)
            next_action_samples = self.vae.decode(next_state_repeat)
            next_action_samples = next_action_samples + self.perturbation(next_state_repeat, next_action_samples)
            q_values = self.q_target(next_state_repeat, next_action_samples)
            q_values = q_values.view(num_samples, batch_size, 1)
            max_q = q_values.max(0)[0]
            target_q = reward + (1 - done) * self.discount * max_q

        current_q = self.q_network(state, action)
        q_loss = nn.MSELoss()(current_q, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Perturbation network update.
        perturbed_actions = self.vae.decode(state)
        perturbed_actions = perturbed_actions + self.perturbation(state, perturbed_actions)
        perturb_loss = -self.q_network(state, perturbed_actions).mean()
        self.perturbation_optimizer.zero_grad()
        perturb_loss.backward()
        self.perturbation_optimizer.step()

        # Soft update target Q-network.
        for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --------------------------
# Main Training Loop
# --------------------------


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "JNJ", "V", "PG", "MA",
               "NVDA", "UNH", "HD", "DIS", "BAC", "PFE", "CMCSA", "VZ", "ADBE", "NFLX"]
    
    data = yf.download(tickers, start="2015-01-01", end="2022-12-31")["Close"]
    data = data.fillna(method='ffill').fillna(method='bfill')
    stock_prices = data.values.astype(np.float32)

    env = TradingEnv(stock_prices, initial_cash=10000)
    state = env.reset()

    num_episodes = 100
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = np.random.rand(env.num_stocks)  # Placeholder for RL agent action
            action = action / action.sum()  # Ensure it sums to 1
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        final_prices = env.stock_prices[-1]
        portfolio_value = env.cash + np.sum(env.holdings * final_prices)
        episode_rewards.append(total_reward)

        print(f"Episode {episode+1} End: Portfolio Value = ${portfolio_value:.2f}, Cash = ${env.cash:.2f}")

    # Final episode summary
    print("\n===== Final Training Episode Summary =====")
    initial_portfolio_value = 10000.00
    final_portfolio_value = env.cash + np.sum(env.holdings * final_prices)
    total_change = final_portfolio_value - initial_portfolio_value
    percentage_change = (total_change / initial_portfolio_value) * 100

    print(f"Initial Portfolio Value: ${initial_portfolio_value:.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    print(f"Total Change: ${total_change:.2f} ({percentage_change:.2f}%)\n")
    
    print("Stock Breakdown:")
    for i, ticker in enumerate(tickers):
        initial_stock_value = (10000 / env.num_stocks)  # Initial cash divided equally among stocks
        final_stock_value = env.holdings[i] * final_prices[i]
        change = final_stock_value - initial_stock_value
        print(f"{ticker}: Initial ${initial_stock_value:.2f}, Final ${final_stock_value:.2f}, Change ${change:.2f}")

    print(f"Cash at End: ${env.cash:.2f}")

    # Plot rewards
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Portfolio Increase)")
    plt.title("BCQ Training on Trading Environment (2015-2022)")
    plt.show()