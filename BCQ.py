import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

# --------------------------
# Trading Environment
# --------------------------
class TradingEnv(gym.Env):
    """
    Trading environment with daily rebalancing and debug logs.
    
    - Starts with $10,000 equally invested in 20 stocks.
    - Uses daily closing prices from yfinance.
    - At each step, the agent supplies a target allocation vector (length=num_stocks+1) 
      that sums to 1. The first num_stocks entries specify target allocation for stocks,
      and the last entry specifies allocation for cash.
    - The environment rebalances by selling or buying stocks according to the target.
    - The portfolio value is updated every day according to the current closing prices.
    - Reward is computed as the increase in portfolio value over the day.
    - Debug logs print the day number, portfolio value, and cash balance.
    """
    def __init__(self, stock_prices, dividends, initial_cash=10000):
        super().__init__()
        self.stock_prices = stock_prices  # shape: (T, num_stocks)
        self.dividends = dividends
        self.initial_cash = initial_cash
        self.num_stocks = stock_prices.shape[1]
        self.current_step = 0
        
        # Updated action: now expecting (num_stocks + 1) allocations (stocks + cash).
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks + 1,), dtype=np.float32)
        # Updated observation: prices, holdings for stocks, and cash amount.
        obs_low = np.zeros(self.num_stocks * 2 + 1, dtype=np.float32)
        obs_high = np.full(self.num_stocks * 2 + 1, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self):
        self.current_step = 0
        current_prices = self.stock_prices[self.current_step]
        # Start with equal allocation in stocks and no cash.
        allocation = np.full(self.num_stocks, 1.0 / self.num_stocks)
        self.cash = 0.0  # All cash is used initially to buy stocks.
        self.holdings = (allocation * self.initial_cash) / current_prices
        self.cost_basis = current_prices.copy()
        self.sold_profit = 0.0
        return self._get_obs()

    def _get_obs(self):
        # Observation includes current prices, current holdings, and available cash.
        current_prices = self.stock_prices[self.current_step]
        return np.concatenate([current_prices, self.holdings, [self.cash]])

    def step(self, action):
        # Expecting action vector of size num_stocks + 1; first num_stocks for stocks and last for cash.
        assert len(action) == self.num_stocks + 1, "Action must include allocation for all stocks plus cash."
        
        # Ensure valid probability distribution.
        action = np.clip(action, 1e-6, None)
        action /= np.sum(action)

        # Get current prices.
        current_prices = self.stock_prices[self.current_step]

        # Compute current portfolio value.
        stock_values = self.holdings * current_prices
        stock_total = np.sum(stock_values)
        total_value = stock_total + self.cash

        # Rebalance portfolio according to target allocation.
        target_stock_value = action[:-1] * total_value
        new_holdings = target_stock_value / current_prices  # new shares for each stock
        new_cash = action[-1] * total_value

        # Compute reward using next day's prices.
        next_step = self.current_step + 1
        next_prices = self.stock_prices[next_step]  # assumes there is a next day; done will handle terminal
        next_stock_value = np.sum(new_holdings * next_prices)
        next_total_value = next_stock_value + new_cash
        reward = next_total_value - total_value

        # Update state.
        self.holdings = new_holdings
        self.cash = new_cash
        self.current_step = next_step
        done = self.current_step >= len(self.stock_prices) - 1
        obs = self._get_obs()

        return obs, reward, done, {}

    def render(self, mode="human"):
        # Render uses the previous day's prices (to show the rebalanced portfolio).
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
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim).to(device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)

        self.perturbation = Perturbation(state_dim, action_dim).to(device)
        self.perturbation_optimizer = optim.Adam(self.perturbation.parameters(), lr=1e-3)
        
        self.replay_buffer = ReplayBuffer(100000)
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            num_samples = 10
            state_repeat = state_tensor.repeat(num_samples, 1)
            action_samples = self.vae.decode(state_repeat)
            # Apply perturbation and select the best action based on Q-value.
            perturbed_actions = action_samples + self.perturbation(state_repeat, action_samples)
            q_values = self.q_network(state_repeat, perturbed_actions)
            best_index = q_values.argmax()
            best_action = perturbed_actions[best_index].cpu().numpy()
        # Map from [-1, 1] to [0, 1] and normalize so that the allocation sums to 1.
        best_action = (best_action + 1) / 2
        best_action = best_action / best_action.sum()
        return best_action

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
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
    # Set random seeds for reproducibility
    start = "2005-01-01"
    end = "2024-12-31"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Device configuration: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Stock tickers and data download
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "JNJ", "V", "PG", "MA",
                "NVDA", "UNH", "HD", "DIS", "BAC", "PFE", "CMCSA", "VZ", "ADBE", "NFLX"]
    dividends = {}
    for item in tickers:
        ticker_obj = yf.Ticker(item)
        dividends[item] = pd.Series(ticker_obj.dividends[start:end].values.astype(np.float32)).fillna(method='ffill')
    print("Downloading stock data...")
    data = yf.download(tickers, start=start, end=end)["Close"]
    data = data.fillna(method='ffill').fillna(method='bfill')
    stock_prices = data.values.astype(np.float32)
    
    # Create environment
    env = TradingEnv(stock_prices, dividends, initial_cash=10000)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize BCQ agent
    agent = BCQAgent(state_dim, action_dim, device)
    
    # Training parameters
    num_episodes = 10
    batch_size = 32
    print_interval = 50
    episode_rewards = []
    portfolio_values = []
    
    print("Starting training...")
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train the agent
            agent.train(batch_size)
            
            state = next_state
            total_reward += reward
            
            # Optional: stop early for debugging
            if env.current_step == 2000:
                print(f"Episode {episode} finishing early")
                break
        
        # Record results
        final_prices = env.stock_prices[-1]
        portfolio_value = env.cash + np.sum(env.holdings * final_prices)
        episode_rewards.append(total_reward)
        portfolio_values.append(portfolio_value)
        
        # Print progress
        if episode % print_interval == 0:
            print(f"Episode {episode}/{num_episodes} - "
                  f"Reward: ${total_reward:.2f} - "
                  f"Portfolio Value: ${portfolio_value:.2f} - "
                  f"Cash: ${env.cash:.2f}")
    
    # Training complete
    print("\nTraining completed!")
    
    # Final episode summary
    print("\n===== Final Training Episode Summary =====")
    initial_portfolio_value = 10000.00
    final_portfolio_value = portfolio_values[-1]
    total_change = final_portfolio_value - initial_portfolio_value
    percentage_change = (total_change / initial_portfolio_value) * 100

    print(f"Initial Portfolio Value: ${initial_portfolio_value:.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    print(f"Total Change: ${total_change:.2f} ({percentage_change:.2f}%)\n")
    
    print("Stock Breakdown:")
    for i, ticker in enumerate(tickers):
        initial_stock_value = (10000 / env.num_stocks)  # equally distributed initial cash
        final_stock_value = env.holdings[i] * final_prices[i]
        change = final_stock_value - initial_stock_value
        print(f"{ticker}: Initial ${initial_stock_value:.2f}, Final ${final_stock_value:.2f}, Change ${change:.2f}")

    print(f"Cash at End: ${env.cash:.2f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot portfolio values
    plt.subplot(1, 2, 1)
    plt.plot(portfolio_values)
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Portfolio Value Over Episodes")
    
    # Plot rewards
    plt.subplot(1, 2, 2)
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward ($)")
    plt.title("Rewards Over Episodes")
    
    plt.tight_layout()
    plt.show() 

    # --------------------------
    # Plot daily performance of final episode
    # --------------------------
    print("\nPlotting portfolio value over the last episode...")
    daily_values = []
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        current_prices = env.stock_prices[env.current_step - 1]
        daily_value = env.cash + np.sum(env.holdings * current_prices)
        daily_values.append(daily_value)

    plt.figure(figsize=(10, 5))
    plt.plot(daily_values)
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Final Episode: Daily Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
