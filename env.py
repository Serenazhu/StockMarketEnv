import gymnasium as gym
from gymnasium import spaces
import numpy as np
"""
This script implements a custom stock market environment using OpenAI's Gym library for reinforcement learning. 
The environment simulates buying and selling shares based on random stock prices and tracks metrics such as 
current balance, number of shares, total cost, and average cost per share.

Key features:
- Action space: Consists of buying or selling shares.
- Observation space: Tracks stock price, balance, shares held, and average cost.
- Step function: Processes buy/sell actions, updates stock price, and calculates rewards based on profits or losses.
- Reset function: Initializes the environment at the start of each episode.

The environment is designed to be used for training RL agents to develop stock trading strategies.
"""

class StockMarketEnv(gym.Env):
    def __init__(self, init_balance=1000, init_shares=10):
        super().__init__()
        self.init_balance = init_balance  
        self.init_shares = init_shares
        self.current_balance = self.init_balance  
        self.num_shares = self.init_shares
        self.stock_data = {
            "price": np.random.randint(100, 301)
        }
        self.total_cost = 0
        self.average_cost = 0

        # Initialize action space
        self.action_space = spaces.Tuple((
            spaces.Box(low=0, high=self.current_balance, shape=(1,), dtype=np.float32), # Amount to buy ($)
            spaces.Box(low=self.init_shares, high=self.num_shares, shape=(1,), dtype=np.float32) # Amount to sell (# of shares)
        ))
        
        # TODO: Needs to be updated to include all the stock data
        self.observation_space = spaces.Dict({
            "price": spaces.Box(low=100, high=300, shape=(1,), dtype=np.float32),
            "current_balance": spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            "num_shares": spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            "average_cost": spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32)
        })
        
    # Calls at the start of each episode
    def reset(self):
        self.current_balance = self.init_balance  
        self.num_shares = self.init_shares
        self.total_cost = 1000
        self.average_cost = 1000
        return self._get_obs()

    def _get_obs(self):
        return {
            "price": np.array([self.stock_data["price"]], dtype=np.float32),
            "current_balance": np.array([self.current_balance], dtype=np.float32),
            "num_shares": np.array([self.num_shares], dtype=np.float32),
            "average_cost": np.array([self.average_cost], dtype=np.float32)
        }
    

    def step(self, action):
        # Unpack the action
        buy_amount, sell_amount = action
        
        # Get the current stock price
        current_price = self.stock_data["price"]
        
        # Initialize reward
        reward = 0
        
        # Execute the action
        if buy_amount > 0 and buy_amount[0] <= self.current_balance:  # Buy
            shares_to_buy = buy_amount[0] / current_price
            print("Current balance: ", self.current_balance)
            print("Current shares: ", self.num_shares)
            print("Price: ", current_price)
            print("Bought: ", shares_to_buy, "shares")
            
            # Deduct the cost from the current balance
            self.current_balance -= buy_amount[0]   
            print("Current balance: ", self.current_balance)
            
            # Update the number of shares
            self.num_shares += shares_to_buy
            print("Number of shares: ", self.num_shares)
            
            # Update the total cost by adding the cost of the new shares
            new_shares_cost = shares_to_buy * current_price
            self.total_cost += new_shares_cost
            print("Total cost: ", self.total_cost)
            
            # Recalculate the average cost per share
            self.average_cost = self.total_cost / self.num_shares
            print("Average cost: ", self.average_cost)
            
            print("BOUGHT: ", shares_to_buy, "shares")
            print("--------------------------------")

        if sell_amount > 0 and sell_amount[0] <= self.num_shares:  # Sell
                shares_to_sell = sell_amount[0]
                print("Current balance: ", self.current_balance)
                print("Current shares: ", self.num_shares)
                print("Price: ", current_price)
                print("Sold: ", shares_to_sell, "shares")
                
        # Calculate the revenue from selling shares
                revenue = shares_to_sell * current_price
                print("Revenue: ", revenue)
                
                # Add the revenue to the current balance
                self.current_balance += revenue
                print("Current balance: ", self.current_balance)
                
                # Reduce the number of shares
                self.num_shares -= shares_to_sell
                
                # Adjust the total cost by deducting the cost of the sold shares (using the average cost)
                self.total_cost -= shares_to_sell * self.average_cost
                print("Total cost: ", self.total_cost)
                
                print("Number of shares: ", self.num_shares)
                
                # Calculate profit or loss from the sale
                profit_loss = shares_to_sell * (current_price - self.average_cost)
                print("Profit/Loss: ", profit_loss)
                
                # Add the profit/loss to the reward
                reward += profit_loss
                print("Reward: ", reward)
                
                print("SOLD: ", shares_to_sell, "shares")
                print("--------------------------------")

            # Update the stock price 
                self.stock_data["price"] += np.random.randint(-10, 11)
                self.stock_data["price"] = max(100, min(300, self.stock_data["price"]))  # Ensure price stays within bounds
            
        # Check if episode is done 
        done = self.current_balance <= 0 and self.num_shares <= 0 or self.current_balance >= 2000
            
        # Get the new observation
        obs = self._get_obs()
            
        return obs, reward, done, False, {}  # The False is for the 'truncated' flag, {} for info

env = StockMarketEnv()

# Reset the environment to start a new episode
obs = env.reset()

done = False


while not done:
    # Sample a random action
    action = env.action_space.sample()
    
    # Take the action and get the new state, reward, and whether the game is done
    obs, reward, done, truncated, info = env.step(action)
    print(f"obs: {obs}")
    print("--------------------------------")
    print(f"reward: {reward}")
    print("--------------------------------")

# Close the environment when done
env.close()
