# **Delivery Optimization README**

## **Overview**
This project implements an AI-driven delivery optimization model using a hybrid approach of Integer Linear Programming (ILP) and Reinforcement Learning (RL). It dynamically assigns riders, optimizes routes, and enhances efficiency in real-time delivery scenarios.

## **Installation & Setup**

```bash
# 1. Install required dependencies
pip install stable-baselines3[extra] gym osrm folium pandas scikit-learn torch

# 2. Run the training script (save as delivery_rl.py)
python delivery_rl.py

# 3. For real-time monitoring (in a separate terminal):
tensorboard --logdir ./logs

# 4. To see visualizations after training:
open delivery_routes.html  # On Windows: start delivery_routes.html
```

## **Key Terminal Commands & Expected Outputs**

### **1. Training Progress**
```bash
python delivery_rl.py
```
**Expected Output:**
```
Using cpu device
Creating environment...
| rollout/            |          |
|    ep_len_mean      | 150      |
|    ep_rew_mean      | -12.4    |
|    exploration_rate | 0.95     |
| time/               |          |
|    episodes         | 100      |
|    fps              | 45       |
|    time_elapsed     | 60       |
|    total_timesteps  | 15000    |
```

### **2. Tensorboard Monitoring**
```bash
tensorboard --logdir ./logs
```
**Access:** `http://localhost:6006/` to view:
- Episode Reward Progression
- Training Loss Curves
- Exploration Rate Decay

### **3. Visualization Output**
```bash
open delivery_routes.html
```
**Will show:**
- Interactive map with:
  - Blue circles: Pending orders
  - Green markers: Active riders
  - Red lines: Late delivery routes
  - Yellow lines: Optimized paths

## **For Quick Testing (Reduced Scale):**
```bash
# Run with smaller parameters
python delivery_rl.py --num_riders 10 --max_orders 50 --timesteps 10000
```

## **Troubleshooting Common Issues**

### **1. OSRM Connection Errors:**
```bash
# Start local OSRM instance (requires Docker)
docker run -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/india-latest.osrm
```

### **2. CUDA/MPS Acceleration:**
```bash
# For GPU training (NVIDIA)
python delivery_rl.py --device cuda

# For Apple Silicon
python delivery_rl.py --device mps
```

### **3. Real-time Debugging:**
```bash
# Add debug flag
python delivery_rl.py --debug

# Sample debug output:
[DEBUG] Assigned order 4512 to rider 38
[DEBUG] New route saved 5.2km (Prev: 6.7km)
[DEBUG] Peak hour penalty applied at 13:00
```

## **Expected Performance Metrics**
- Initial Runs: Negative rewards (-20 to -5 range)
- After 50k Steps: Positive rewards (+5 to +20)
- Final Model: Avg Reward >15 with 85%+ On-Time Deliveries

**Note:** Training will take 30+ minutes for 100k timesteps on CPU. Use `--timesteps 1000` for quick verification.



