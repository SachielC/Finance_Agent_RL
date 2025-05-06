## **RL for Offline Stock Trading**

Johnathan Finizio and Sachiel Chuckrow

### **Problem Statement:** How can we effectively use reinforcement learning to determine optimal stock trading behaviors? 

### **Proposed RL Techniques**

**Existing Projects:**  
**RL projects relating to stock analysis:**

* **Stock Price Predictions Using Reinforcement Learning  |**  2001  |  *Jae Won Lee* 

[https://ieeexplore.ieee.org/document/931880?denied=](https://ieeexplore.ieee.org/document/931880?denied=)

* An older paper \- could provide understanding on the basic approach to the problem, rather than to directly implement a similar model  
  * States how the nature of stock rewards is more suitable for reinforcement learning \- contains immediate and delayed rewards  
  * Models stock price changes as a Markov process, and uses TD Learning  
* **Application of Deep Reinforcement Learning in Stock Trading Strategies and Stock Forecasting  |**  2019  | *Li et al.*  
  [Application of deep reinforcement learning in stock trading strategies and stock forecasting | Computing](https://link.springer.com/article/10.1007/s00607-019-00773-w)   
  * Implements deep reinforcement learning as the main approach (in this case DQN) \- combines the perception and feature extraction abilities of deep learning and the decision making abilities of reinforcement learning (similar to naive blackjack homework in terms of strategies)  
  * However, uses online learning which isn’t the best approach \- provides good reason to stick with offline RL

**Rl projects relating to offline learning:**

* **Conservative Q-Learning for Offline Reinforcement Learning**  |  2020  |  *Kumar et al.*

[\[2006.04779\] Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779) 

* Offline RL algorithms can suffer from action distribution shift \- 𝜋 may be biased towards out of distributions actions with erroneously high Q-values  
  * In standard Rl, this can be corrected by trying an action in an environment and observing the value, but since we can’t interact with the environment in our cost (too costly) this is not possible  
  * The CQL framework calculates the expected value of a policy under the learned Q function so that is lower bounds its true value (this prevents overestimation)  
  * Instead of fully evaluating each policy iteration, the algorithm uses the Q function to approximate the best value at each step  
* **Constrained Q-Learning for Batch Process Optimization**  |  2021  | *Pan et al.*  
  [https://www.sciencedirect.com/science/article/pii/S2405896321010636](https://www.sciencedirect.com/science/article/pii/S2405896321010636)   
  * Proposes an “oracle”-assisted constrained Q-learning algorithm  
  * Uses safety constraints suitable for “safety critical” circumstances  
  * Dynamically adjusts buffer margins that allows for self-tuning

**RL techniques we will use:**  
Offline RL makes the most sense for this challenge since it allows us to experiment with a lesser cost, and much faster, as opposed to working in real time. Using a variety of models and comparing them will be ideal here. Conservative Q-learning and Batch-Constrained Q-learning are the optimal approaches for our offline RL problem. They minimize overestimation, provide safe and conservative actions, and are more robust to shifts in the market (the policy will not drastically shift). Overall, these approaches help to mitigate the risks that DQN poses in online learning and are well suited for our goal.   
However,  Li et al. stated that Naive DQNs may be a feasible starting model, despite being online learning. Experimentation with PPOs and ensemble methods (using expert data if available?) may prove to be useful as well. We can implement these models using environments in the FinRL library, in order to use online learning.  
Lastly, beyond RL, we will test traditional supervised learning models to compare to our RL algorithms performance.

### **Expected Challenges**

* Ensuring that our model balances profitability and risk. This could get out of hand if our model overfits to the data  
* Gathering sufficient and consistent data throughout time periods (consistent with companies and metrics).

### **Datasets and Environments** 

* **Kaggle Dataset**  
  [https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data](https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data)  
  * Has historic data on over 7000 companies  
  * Could be difficult to clean data, given that time periods are different and each stock is in its on .txt file  
* **FinRL**  |  2022  | *Liu et al.*   
  [\[2011.09607\] FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance](https://arxiv.org/abs/2011.09607)  
  * Deep reinforcement learning library for automated stock trading  
  * Can be utilized for benchmark tests, single stock trading, multi stock trading, portfolio allocation, and user-defined trading tasks  
  * Can implement both conventional RL agents and DRL agents  
  * Environments include: benchmark environment, various constituent environments, user imported dataset  
* **Yahoo Finance**  
  * Library downloadable into python (yfinance)  
  * Includes stock data as well as Dividends. Those are reported intermittently but basic data cleaning can allow the dataset to look backwards .  

### **Evaluation Metrics**

To evaluate our model, we need to evaluate both the profitability and the risk associated with our policy. Here are some metrics we may use:

* **Profitability Metrics**  
  * Cumulative Return (total rewards)  
  * Annualized Return (annual rewards)  
  * Profit Factor (total profit / total loss)  
* **Risk Metrics**  
  * Maximum Drawdown (largest peak to valley loss in period)  
  * Win Rate % (percentage of profitable trades)  
  * Average Trade Return
