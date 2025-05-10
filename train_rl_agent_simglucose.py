import os
import numpy as np
import torch
from datetime import datetime, timedelta
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.base import Controller, Action
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim

# Add RL_V3 to Python path
import sys
sys.path.append(os.path.join(os.getcwd(), 'RL_V3'))
from glucose_controller_rl.controller.pid_controller_v3 import PIDControllerV3
from glucose_controller_rl.agent.rl_agent_v3 import RLAgentV3
from glucose_controller_rl.utils import config_v3 as cfg

class RLController(Controller):
    def __init__(self, init_state=None):
        super().__init__(init_state)
        self.pid = PIDControllerV3()
        self.agent = RLAgentV3()
        self.last_time = None
        self.last_glucose = None
        self.meal_flag = 0
        self.time_since_meal = 0
        self.start_time = None
        self.training = True  # Flag to control exploration during training
    
    def _get_state(self, current_glucose, current_time_min):
        if self.last_glucose is None:
            norm_error = 0
            norm_trend = 0
        else:
            error = current_glucose - cfg.TARGET_GLUCOSE
            norm_error = error / cfg.STATE_ERROR_NORM_FACTOR
            trend = (current_glucose - self.last_glucose) / (current_time_min - self.last_time) if self.last_time is not None else 0
            norm_trend = trend / cfg.STATE_TREND_NORM_FACTOR
        
        # Update meal flag and time since meal
        if self.meal_flag > 0:
            self.time_since_meal += 1
            if self.time_since_meal >= cfg.STATE_TIME_SINCE_MEAL_HORIZON:
                self.meal_flag = 0
                self.time_since_meal = 0
        
        return np.array([norm_error, norm_trend, self.meal_flag], dtype=np.float32)
    
    def policy(self, observation, reward, done, **info):
        # Extract CGM value from observation (Step object)
        if hasattr(observation, 'observation'):
            current_glucose = observation.observation.CGM
        else:
            current_glucose = observation.CGM
        
        current_time = info.get('time')
        meal = info.get('meal', 0)
        
        # Convert datetime to minutes since start
        if self.start_time is None:
            self.start_time = current_time
        current_time_min = int((current_time - self.start_time).total_seconds() / 60)
        
        # Update state
        state = self._get_state(current_glucose, current_time_min)
        
        # Get gains from agent with exploration during training
        gains = self.agent.get_gains(state, explore=self.training)
        self.pid.set_gains(gains[0], gains[1], gains[2], 
                          (cfg.AGENT_KP_MIN, cfg.AGENT_KI_MIN, cfg.AGENT_KD_MIN),
                          (cfg.AGENT_KP_MAX, cfg.AGENT_KI_MAX, cfg.AGENT_KD_MAX))
        
        # Get insulin rate from PID
        insulin_rate = self.pid.update(current_glucose, current_time_min)
        
        # Update state for next step
        self.last_time = current_time_min
        self.last_glucose = current_glucose
        
        # Check for meals
        if meal > 0:
            self.meal_flag = 1
            self.time_since_meal = 0
        
        return Action(basal=insulin_rate, bolus=0)
    
    def reset(self):
        self.pid.reset()
        self.last_time = None
        self.last_glucose = None
        self.meal_flag = 0
        self.time_since_meal = 0
        self.start_time = None

def create_meal_scenario():
    """Create a random meal scenario for training."""
    start_time = datetime.combine(datetime.now().date(), datetime.min.time())
    scenario = []
    
    # Add 3-4 random meals
    num_meals = np.random.randint(3, 5)
    meal_times = sorted(np.random.choice(range(6, 22), num_meals, replace=False))
    meal_sizes = np.random.randint(30, 76, num_meals)
    
    for time, size in zip(meal_times, meal_sizes):
        scenario.append((float(time), float(size)))  # Convert to float hours
    
    return CustomScenario(start_time=start_time, scenario=scenario)

def calculate_reward(bg_history, cgm_history, insulin_history, cho_history):
    """Calculate reward based on glucose control metrics."""
    bg = np.array(bg_history)
    cgm = np.array(cgm_history)
    insulin = np.array(insulin_history)
    cho = np.array(cho_history)
    
    # Time in range reward (70-180 mg/dL)
    tir = np.mean((bg >= 70) & (bg <= 180))
    tir_reward = 100 * tir
    
    # Hypoglycemia penalty
    hypo_penalty = -100 * np.mean(bg < 70)
    
    # Hyperglycemia penalty
    hyper_penalty = -50 * np.mean(bg > 180)
    
    # Variability penalty
    variability_penalty = -10 * np.std(bg)
    
    # Total reward
    reward = tir_reward + hypo_penalty + hyper_penalty + variability_penalty
    
    return reward

def train_episode(env, controller, episode_num):
    """Run one training episode."""
    # Reset environment and controller
    obs = env.reset()
    controller.reset()
    controller.training = True
    
    # Initialize episode data
    bg_history = []
    cgm_history = []
    insulin_history = []
    cho_history = []
    rewards = []
    
    # Run episode
    done = False
    info = {'time': env.scenario.start_time, 'meal': 0}  # Initialize info
    step_count = 0
    max_steps = 1440  # 24 hours * 60 minutes
    
    while not done and step_count < max_steps:
        # Get action from controller
        action = controller.policy(obs, reward=0, done=done, time=env.time, meal=info.get('meal', 0))
        
        # Take step in environment
        obs, reward, done, info = env.step(action)
        
        # Store data
        bg_history.append(info['bg'])
        if hasattr(obs, 'observation'):
            cgm_history.append(obs.observation.CGM)
        else:
            cgm_history.append(obs.CGM)
        insulin_history.append(action.basal)
        cho_history.append(info['meal'])
        rewards.append(reward)
        
        step_count += 1
        if step_count % 60 == 0:  # Print progress every hour
            print("Episode {} - Step {} - Current BG: {:.1f} mg/dL".format(episode_num, step_count, info['bg']))
    
    # Calculate final reward
    episode_reward = calculate_reward(bg_history, cgm_history, insulin_history, cho_history)
    
    # Print episode summary
    print("\nEpisode {} Summary:".format(episode_num))
    print("Total Steps: {}".format(step_count))
    print("Mean BG: {:.1f} mg/dL".format(np.mean(bg_history)))
    print("Time in Range: {:.1f}%".format(100 * np.mean((np.array(bg_history) >= 70) & (np.array(bg_history) <= 180))))
    print("Episode Reward: {:.1f}".format(episode_reward))
    if step_count >= max_steps:
        print("Episode ended due to reaching maximum steps (24 hours)")
    print("----------------------")
    
    return episode_reward

def main():
    # Create results directory
    results_path = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_path, exist_ok=True)
    
    # Initialize components
    patient = T1DPatient.withName('adolescent#001')
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    controller = RLController()
    
    # Training parameters
    num_episodes = 1000
    best_reward = float('-inf')
    checkpoint_interval = 10  # Save checkpoint every 10 episodes
    
    print("\nStarting Training...")
    print("Total Episodes: {}".format(num_episodes))
    print("Checkpoint Interval: Every {} episodes".format(checkpoint_interval))
    print("Press Ctrl+C to stop training and save the model")
    print("----------------------")
    
    try:
        # Training loop
        for episode in range(num_episodes):
            # Create new meal scenario for each episode
            scenario = create_meal_scenario()
            
            # Create environment
            env = T1DSimEnv(patient, sensor, pump, scenario)
            
            # Run episode
            episode_reward = train_episode(env, controller, episode)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                controller.agent.save(cfg.get_best_agent_path())
                print("New best model saved with reward: {:.1f}".format(best_reward))
            
            # Save checkpoint every checkpoint_interval episodes
            if (episode + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(results_path, f'checkpoint_episode_{episode + 1}.pt')
                controller.agent.save(checkpoint_path)
                print("\nCheckpoint saved at episode {}: {}".format(episode + 1, checkpoint_path))
                print("Current best reward: {:.1f}".format(best_reward))
                print("----------------------")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        # Save final model
        final_path = os.path.join(results_path, 'final_model.pt')
        controller.agent.save(final_path)
        print("Final model saved at: {}".format(final_path))
        print("Best reward achieved: {:.1f}".format(best_reward))
    
    print("\nTraining completed!")
    print("Best reward achieved: {:.1f}".format(best_reward))
    print("Final model saved at: {}".format(os.path.join(results_path, 'final_model.pt')))

if __name__ == "__main__":
    main() 