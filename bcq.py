import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# --------------------------
# Trading Environment
# --------------------------
class TradingEnv(gym.Env):
    """
    Trading environment where you start with $10,000 and must fully invest
    in stocks at the first step according to an allocation vector.
    After the initial purchase, no selling is allowed.
    The observation consists of the current stock prices and your current holdings.
    """
    def __init__(self, stock_prices, initial_cash=10000):
        super().__init__()
        self.stock_prices = stock_prices  # shape: (T, num_stocks)
        self.initial_cash = initial_cash
        self.num_stocks = stock_prices.shape[1]
        self.current_step = 0
        
        # Action: allocation vector (continuous values); agent must output vector that sums to 1.
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)
        # Observation: concatenation of current prices and holdings.
        obs_low = np.zeros(self.num_stocks * 2, dtype=np.float32)
        obs_high = np.full(self.num_stocks * 2, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.initialized = False

    def reset(self):
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.num_stocks)  # no stocks initially
        self.current_step = 0
        self.initialized = False
        return self._get_obs()
    
    def _get_obs(self):
        current_prices = self.stock_prices[self.current_step]
        return np.concatenate([current_prices, self.holdings])
    
    def step(self, action):
        """
        At time 0, use the action (allocation vector) to invest all cash.
        Afterwards, no trading (i.e. no selling) is allowed.
        """
        if self.current_step == 0 and not self.initialized:
            # Normalize allocation so it sums to 1.
            allocation = np.array(action) / np.sum(action)
            current_prices = self.stock_prices[self.current_step]
            # Buy shares: number of shares = (cash * allocation) / price.
            self.holdings = (self.cash * allocation) / current_prices
            self.cash = 0
            self.initialized = True
            reward = 0  # No reward on initial purchase.
        else:
            # In this simple version, no further trading is allowed.
            reward = 0
        
        self.current_step += 1
        done = self.current_step >= len(self.stock_prices)
        obs = self._get_obs()
        info = {}
        return obs, reward, done, info

    def render(self, mode="human"):
        current_prices = self.stock_prices[self.current_step - 1]
        portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        print(f"Step: {self.current_step} | Portfolio Value: ${portfolio_value:.2f}")

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
# Q-network: takes state and action, outputs Q-value.
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# VAE network: learns to reconstruct actions given state-action pairs.
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
            z = torch.randn(batch_size, self.latent_dim).to(state.device).clamp(-0.5,0.5)
        x = torch.cat([state, z], dim=1)
        x = torch.relu(self.d1(x))
        x = torch.relu(self.d2(x))
        action = torch.tanh(self.d3(x))
        return action

# Perturbation network: given state and action, outputs a small adjustment (delta).
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
        # Map network output (assumed in [-1,1]) to [0,1] and normalize to sum to 1.
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

        # VAE update: reconstruction and KL loss.
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
def main():
    # Create a simulated market: 3 stocks over T time steps.
    T = 200
    num_stocks = 3
    np.random.seed(42)
    base_prices = np.array([100, 50, 20])
    price_changes = np.random.normal(0, 1, size=(T, num_stocks))
    stock_prices = np.maximum(base_prices + np.cumsum(price_changes, axis=0), 1)
    
    env = TradingEnv(stock_prices, initial_cash=10000)
    state = env.reset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]  # prices + holdings
    action_dim = env.action_space.shape[0]        # allocation vector
    
    agent = BCQAgent(state_dim, action_dim, device)
    replay_buffer = ReplayBuffer(100000)
    
    num_episodes = 1000
    batch_size = 64
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0  # In this simple setup, reward is always 0.
        while not done:
            # Use BCQ agent to select an action.
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        
        # Train the agent if enough samples have been collected.
        if len(replay_buffer) > batch_size:
            for _ in range(50):
                agent.train(replay_buffer, batch_size)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward}")
    
    # Plot episode rewards (note: rewards are zero in this example as reward function is not defined).
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("BCQ Training on Trading Environment")
    plt.show()

if __name__ == "__main__":
    main()
