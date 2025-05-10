import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.base import Controller, Action
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim

# Add RL_V3 to Python path
sys.path.append(os.path.join(os.getcwd(), 'RL_V3'))
from glucose_controller_rl.controller.pid_controller_v3 import PIDControllerV3
from glucose_controller_rl.agent.rl_agent_v3 import RLAgentV3
from glucose_controller_rl.utils import config_v3 as cfg

# Create results directory if it doesn't exist
results_path = os.path.join(os.getcwd(), 'results')
os.makedirs(results_path, exist_ok=True)

# 1. Define custom meal scenario
start_time = datetime.combine(datetime.now().date(), datetime.min.time())
scenario = [
    (8, 40),    # 8:00 AM, 40g
    (13, 50),   # 1:00 PM, 50g
    (19, 75),   # 7:00 PM, 75g
]
custom_scenario = CustomScenario(start_time=start_time, scenario=scenario)

# 2. Patient, sensor, pump
patient = T1DPatient.withName('adolescent#001')  # Closest to a 75kg male in default set
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')

# 3. RL Controller setup
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
        
        # Load best trained agent
        best_agent_path = cfg.get_best_agent_path()
        if os.path.exists(best_agent_path):
            print(f"Loading best trained agent from {best_agent_path}")
            self.agent.load(best_agent_path)
        else:
            print("No trained agent found, using default gains")
    
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
        current_glucose = observation.CGM
        current_time = info.get('time')
        meal = info.get('meal', 0)
        
        # Convert datetime to minutes since start
        if self.start_time is None:
            self.start_time = current_time
        current_time_min = int((current_time - self.start_time).total_seconds() / 60)
        
        # Update state
        state = self._get_state(current_glucose, current_time_min)
        
        # Get gains from agent
        gains = self.agent.get_gains(state, explore=False)
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

controller = RLController()

# 4. Environment and simulation
env = T1DSimEnv(patient, sensor, pump, custom_scenario)
sim_time = timedelta(hours=24)
sim_inst = SimObj(env, controller, sim_time, path=results_path)
results = sim(sim_inst)

# Print column names to see available data
print("\nAvailable data columns:")
print(results.columns.tolist())

# 5. Plotting and stats
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(results['BG'], label='Blood Glucose', color='blue')
plt.plot(results['CGM'], label='CGM', color='green', linestyle='--')
plt.axhline(y=70, color='r', linestyle='-', alpha=0.5)
plt.axhline(y=180, color='r', linestyle='-', alpha=0.5)
plt.ylabel('Glucose Level (mg/dL)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(results['insulin'], label='Insulin (U/min)', color='blue')
plt.plot(results['CHO'], label='Carbs (g)', color='red')
plt.ylabel('Insulin (U/min) / Carbs (g)')
plt.xlabel('Time (minutes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nSimulation Statistics:")
print("----------------------")
print("Mean Blood Glucose: {:.1f} mg/dL".format(np.mean(results['BG'])))
print("Standard Deviation: {:.1f} mg/dL".format(np.std(results['BG'])))
print("Min Blood Glucose: {:.1f} mg/dL".format(np.min(results['BG'])))
print("Max Blood Glucose: {:.1f} mg/dL".format(np.max(results['BG'])))
print("Time in Range (70-180 mg/dL): {:.1f}%".format(
    100 * np.mean((np.array(results['BG']) >= 70) & (np.array(results['BG']) <= 180))
))
print("Total Insulin Delivered: {:.1f} U".format(np.sum(results['insulin'])))
print("Total Carbs Consumed: {:.1f} g".format(np.sum(results['CHO']))) 