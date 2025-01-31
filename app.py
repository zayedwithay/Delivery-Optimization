
import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import osrm
import torch
import folium
from datetime import datetime, timedelta

# Constants
LUCKNOW_BOUNDS = {
    'min_lat': 26.50,
    'max_lat': 27.17,
    'min_lon': 80.50,
    'max_lon': 81.22
}
SPEED_PROFILES = {
    'first_mile': 22.5,  # avg of 20-25 km/h
    'last_mile': 13.5,   # avg of 12-15 km/h
    'return_mile': 32.5  # avg of 30-35 km/h
}

class DeliveryEnvironment(gym.Env):
    def __init__(self, num_riders=50, max_orders=200):
        super(DeliveryEnvironment, self).__init__()
        
        # State space: Normalized values [riders, orders, time, positions]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(num_riders * 4 + max_orders * 5 + 1,), dtype=np.float32
        )
        
        # Action space: Assign, Transfer, Delay, Reroute, Complete
        self.action_space = gym.spaces.Discrete(5)
        
        # Initialize OSRM client
        self.osrm_client = osrm.Client(host='http://router.project-osrm.org')
        
        # Rider and order storage
        self.riders = []
        self.orders = []
        self.current_time = datetime.now()
        
    def _generate_synthetic_orders(self, num_orders):
        """Generate synthetic order data within Lucknow bounds"""
        orders = []
        for _ in range(num_orders):
            order = {
                'id': np.random.randint(1000, 9999),
                'lat': np.random.uniform(LUCKNOW_BOUNDS['min_lat'], LUCKNOW_BOUNDS['max_lat']),
                'lon': np.random.uniform(LUCKNOW_BOUNDS['min_lon'], LUCKNOW_BOUNDS['max_lon']),
                'weight': np.random.choice([0.5, 1.0, 1.5], p=[0.6, 0.3, 0.1]),
                'prep_time': np.random.randint(10, 30),
                'order_time': self.current_time
            }
            orders.append(order)
        return pd.DataFrame(orders)
    
    def _cluster_orders(self, orders):
        """DBSCAN clustering with capacity constraints"""
        coords = orders[['lat', 'lon']].values
        db = DBSCAN(eps=0.5/111, min_samples=2, metric='euclidean')  # ~0.5 km
        clusters = db.fit_predict(coords)
        orders['cluster'] = clusters
        
        # Split clusters exceeding capacity
        valid_clusters = []
        for cluster_id in orders['cluster'].unique():
            cluster_orders = orders[orders['cluster'] == cluster_id]
            total_weight = cluster_orders['weight'].sum()
            if total_weight <= 15:  # 15 kg buffer
                valid_clusters.append(cluster_id)
            else:
                orders.loc[cluster_orders.index, 'cluster'] = -1
        return orders[orders['cluster'] != -1]

    def _calculate_eta(self, order):
        """Calculate ETA considering all parameters"""
        # Get route from restaurant to customer
        route = self.osrm_client.route(
            coordinates=[[order['lon'], order['lat']]],
            overview=False
        )
        distance = route['routes'][0]['distance'] / 1000  # km
        
        # Calculate last mile time
        last_mile_time = (distance / SPEED_PROFILES['last_mile']) * 60
        
        # Total ETA with buffers
        return last_mile_time + order['prep_time'] + 5  # minutes

    def _get_state(self):
        """Normalize environment state for RL model"""
        state = []
        
        # Rider states (location, load, active time)
        for rider in self.riders:
            state.extend([
                rider['lat'] / LUCKNOW_BOUNDS['max_lat'],
                rider['lon'] / LUCKNOW_BOUNDS['max_lon'],
                rider['current_load'] / 20,
                rider['active_time'] / 135  # max 2h15m
            ])
        
        # Order states (location, weight, eta, cluster)
        for _, order in self.orders.iterrows():
            state.extend([
                order['lat'] / LUCKNOW_BOUNDS['max_lat'],
                order['lon'] / LUCKNOW_BOUNDS['max_lon'],
                order['weight'] / 20,
                order['eta'] / 120,  # max 2h eta
                order['cluster'] / 100  # normalized cluster ID
            ])
        
        # Time of day (peak/non-peak)
        hour = self.current_time.hour
        is_peak = 1 if (12 <= hour < 15) or (19 <= hour < 23) else 0
        state.append(is_peak)
        
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Execute one environment step"""
        reward = 0
        done = False
        
        # Action implementation
        if action == 0:  # Assign order
            reward += self._assign_order()
        elif action == 1:  # Transfer order
            reward += self._transfer_order()
        elif action == 2:  # Delay assignment
            reward += self._delay_assignment()
        elif action == 3:  # Reroute rider
            reward += self._reroute_rider()
        elif action == 4:  # Complete gig
            done = self._complete_gig()
        
        # Update environment state
        self.current_time += timedelta(minutes=5)
        next_state = self._get_state()
        
        # Check termination (all orders completed or 6 gigs)
        if len(self.orders) == 0 or self.current_time.hour >= 23:
            done = True
            
        return next_state, reward, done, {}

    def reset(self):
        """Reset environment for new episode"""
        self.riders = [{
            'id': i,
            'lat': np.random.uniform(LUCKNOW_BOUNDS['min_lat'], LUCKNOW_BOUNDS['max_lat']),
            'lon': np.random.uniform(LUCKNOW_BOUNDS['min_lon'], LUCKNOW_BOUNDS['max_lon']),
            'current_load': 0,
            'active_time': 0,
            'gigs_completed': 0
        } for i in range(50)]
        
        self.orders = self._generate_synthetic_orders(200)
        self.orders['eta'] = self.orders.apply(self._calculate_eta, axis=1)
        self.orders = self._cluster_orders(self.orders)
        self.current_time = datetime.now().replace(hour=10, minute=0)
        
        return self._get_state()

    # Helper methods for actions
    def _assign_order(self):
        """Assign optimal order to rider"""
        # Implement order-rider matching logic
        return 10  # Placeholder reward

    def _calculate_reward(self, action):
        """Calculate reward based on action"""
        # Implement comprehensive reward logic
        return {
            0: 5,   # Assign
            1: -2,  # Transfer
            2: -1,  # Delay
            3: -3,  # Reroute
            4: 10   # Complete
        }.get(action, 0)

    def visualize_routes(self):
        """Visualize routes using Folium and OSRM"""
        m = folium.Map(location=[26.8467, 80.9462], zoom_start=12)
        
        # Plot orders
        for _, order in self.orders.iterrows():
            folium.CircleMarker(
                location=[order['lat'], order['lon']],
                radius=3,
                color='blue'
            ).add_to(m)
        
        # Plot riders
        for rider in self.riders:
            folium.Marker(
                location=[rider['lat'], rider['lon']],
                icon=folium.Icon(color='green')
            ).add_to(m)
            
        return m

# Training setup
env = DeliveryEnvironment()
check_env(env)  # Validate environment compatibility

model = DQN(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0003,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99
)

# Train the model
model.learn(total_timesteps=100000)
model.save("delivery_rl_model")

# Example usage
env = DeliveryEnvironment()
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.visualize_routes().save('delivery_routes.html')
    