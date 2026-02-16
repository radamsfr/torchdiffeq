import json
import argparse 
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

import ruckig
from ruckig import InputParameter, Ruckig, Trajectory, Result

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "triple_integrator.json"

class ruckig_generator():
    def __init__(self, config):
        
        # sets variables necesary for ruckig
        self.extract_states(config)
        self.extract_limits(config)
        self.extract_dt(config)
        
        return
    
    def extract_states(self, config: Dict[str, Any]):
        system_cfg = config.get("system", {})

        self.x0 = system_cfg.get("initial_state", [2.0, -2.0, -1.0])  # p0, v0, a0
        self.goal_cfg = system_cfg.get("goal_state", [0.0, 0.0, 0.0])  #pg, vg, ag

        return self.x0, self.goal_cfg
    
    def extract_limits(self, config: Dict[str, Any]):        
        constraints = config.get("constraints", {})
        
        self.limits = {
            "vmax": constraints.get("velocity", [1.0, 1.0])[1],      # use upper bound
            "amax": constraints.get("acceleration", [3.0, 3.0])[1],  # use upper bound
            "jmax": 10.0                                             # you may add this to config later
        }
        
        return self.limits
    
    def extract_dt(self, config:Dict[str, Any]):
        self.dt = config.get("system", {}).get("dt", 0.001)
        
        return self.dt
        

    def build_ruckig_input(self):
        p0, v0, a0 = self.x0
        pg, vg, ag = self.goal_cfg
        limits = self.limits

        inp = InputParameter(1)   # 1 DoF triple integrator

        # Start / goal state (p, v, a)
        inp.current_position = [p0]
        inp.current_velocity = [v0]
        inp.current_acceleration = [a0]

        inp.target_position = [pg]
        inp.target_velocity = [vg]
        inp.target_acceleration = [ag]

        # Limits
        inp.max_velocity = [limits["vmax"]]
        inp.max_acceleration = [limits["amax"]]
        inp.max_jerk = [limits["jmax"]]

        return inp

    def run_ruckig(self, inp):
        # surely this can be cleaned up
        if self.dt:
            dt = self.dt
        else:
            dt = 0.001
        
            
        otg = Ruckig(1, dt)
        trajectory = Trajectory(1)
        
        out_list = []

        result = otg.calculate(inp, trajectory)
        if not result in {Result.Working, Result.Finished}:
            print("Error type:", result.name)
            raise RuntimeError("Ruckig could not calculate trajectory.")

        # sample trajectory over computed duration
        t_samp = np.arange(0, trajectory.duration, dt)

        for t in t_samp:
            p, v, a = trajectory.at_time(t)
            # p, v, a, j = trajectory.at_time(t, return_jerk=True)  #return_jerk not supported in Python ;-;
            
            out_list.append([t, p[0], v[0], a[0]])
        traj = np.array(out_list)
            
        acc = traj[:, 3]
        jerk = np.gradient(acc, t_samp)  # get jerk by getting gradient of acc
        
        traj = np.column_stack((traj, jerk))
        return traj, t_samp

    def plot_trajectory(self, times, traj):
        # time = traj[:, 0] 
        pos = traj[:, 1]
        vel = traj[:, 2]
        acc = traj[:, 3]
        jerk = traj[:, 4]

        # jerk is derivative of acceleration (approx)
        # jerk = np.gradient(acc, times)

        fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

        axs[0].plot(times, pos, label="Position")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(times, vel, label="Velocity")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(times, acc, label="Acceleration")
        axs[2].grid(True)
        axs[2].legend()

        axs[3].plot(times, jerk, label="Jerk")
        axs[3].grid(True)
        axs[3].legend()

        axs[-1].set_xlabel("Time [s]")
        fig.suptitle("Ruckig Time-Optimal Trajectory")
        fig.tight_layout()
        fig.savefig("ruckig_time_optimal_trajectory.png", dpi=300)
        print("Saved: ruckig_time_optimal_trajectory.png")
        
        
def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data

# Parse arguments 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Ruckig Trajectories")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config file.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting the trajectory.",
    )
    return parser.parse_args()
    
def main():
    # print("VERSION:", ruckig.__version__)
    # print(ruckig.__file__)
    
    args = parse_args()
    config = load_config(args.config)
    
    rg = ruckig_generator(config)

    # Extract start and goal, vel and acc limits, dt from config
    start, goal = rg.extract_states(config)
    limits = rg.extract_limits(config)
    dt = rg.extract_dt(config)

    print("Start state:", start)
    print("Goal state:", goal)
    print("Limits:", limits)
    print("dt:", dt)

    # Build Ruckig input
    inp = rg.build_ruckig_input()

    # Run Ruckig
    traj, t = rg.run_ruckig(inp)
    
    # print("traj:\n", traj)
    print("shape:", traj.shape)
    print()

    # Plot results
    rg.plot_trajectory(t, traj)


if __name__ == "__main__":
    main()