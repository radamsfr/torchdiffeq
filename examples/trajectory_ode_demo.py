import os
import argparse
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import ruckig
from ruckig_generator import ruckig_generator, parse_args, load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "triple_integrator.json"
DEFAULT_SAVED_MODEL = Path(__file__).resolve().parents[1] / "models" / "model.pt"


parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4'], default='rk4')
parser.add_argument('--data_size', type=int, default=1000)  # unused
parser.add_argument('--batch_time', type=int, default=70)  # number of time points to sample for each batch (i.e. the length of the trajectory segment used for each training step)
parser.add_argument('--batch_size', type=int, default=30)  # number of batches to sample for each training step (i.e. how many trajectory segments to sample for each training step)
parser.add_argument('--niters', type=int, default=2000)  # number of iterations for training
parser.add_argument('--curriculum_freq', type=int, default=20)  # frequency (in iterations) at which to increase the batch_time (i.e. the length of the trajectory segment used for training), as a form of curriculum learning.
parser.add_argument('--test_freq', type=int, default=20)  # frequency (in iterations) at which to test the model and visualize the trajectory
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--ruckig_config', type=Path, default=DEFAULT_CONFIG_PATH)
parser.add_argument('--load_model', nargs='?', type=Path, const=DEFAULT_SAVED_MODEL, default=None)
parser.add_argument("--save", nargs='?', type=str, const="models", default=None)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()


def get_ruckig_traj(start=None, goal=None, plot_trajectory=False):
    config = load_config(args.ruckig_config)
    rg = ruckig_generator(config)

    if start is None or goal is None:
        start, goal = rg.extract_states(config)

    # print("Ruckig Start state:", start)
    # print("Ruckig Goal state:", goal)
    # print("Ruckig Limits:", limits)
    # print("Ruckig dt:", dt)

    # Build Ruckig input
    inp = rg.build_ruckig_input(start=start, goal=goal)

    # Run Ruckig
    traj, t = rg.run_ruckig(inp)
    
    # print("traj:\n", traj)
    # print("Ruckig shape:", traj.shape)

    # Plot results
    if plot_trajectory:
        rg.plot_trajectory(t, traj)

    return traj, t

def get_random_ruckig_traj(plot_trajectory=False):
    config = load_config(args.ruckig_config)
    rg = ruckig_generator(config)
    
    limits = rg.extract_limits(config)
    dt = rg.extract_dt(config)
    
    buffer = 2.0
        
    random_start_pos = np.random.uniform(-10.0, 10.0)
    random_start_vel = np.random.uniform(-limits['vmax']+buffer, limits['vmax']-buffer)
    random_start_acc = np.random.uniform(-limits['amax']+buffer, limits['amax']-buffer)

    random_goal_pos = np.random.uniform(-10.0, 10.0)
    random_goal_vel = np.random.uniform(-limits['vmax']+buffer, limits['vmax']-buffer)
    random_goal_acc = np.random.uniform(-limits['amax']+buffer, limits['amax']-buffer)

    start = [random_start_pos, random_start_vel, random_start_acc]
    goal = [random_goal_pos, random_goal_vel, random_goal_acc]
    
    # print(f"Ruckig Start state: [{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}]")
    # print(f"Ruckig Goal state: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")
    # print("Ruckig Limits:", limits)
    # print("Ruckig dt:", dt)
    
    # Build Ruckig input with new random start and goal
    inp = rg.build_ruckig_input(start=start, goal=goal)
    
    
    try:
        # Run Ruckig
        traj, t = rg.run_ruckig(inp)
        
    except Exception as e:
        # might fail if random start/goal is not feasible, just return default trajectory
        print(f"Error occurred while running Ruckig: {e}, returning default trajectory")
        traj, t = rg.run_ruckig(rg.build_ruckig_input())
    
        
    if plot_trajectory:
        rg.plot_trajectory(t, traj)

    # Returns (T, 5) where columns are [time, pos, vel, acc, jerk]
    return traj, t

def random_start_goal_pair(limits, buffer=2.0):
    random_start_pos = np.random.uniform(-10.0, 10.0)
    random_start_vel = np.random.uniform(limits['velocity'][0]+buffer, limits['velocity'][1]-buffer)
    random_start_acc = np.random.uniform(limits['acceleration'][0]+buffer, limits['acceleration'][1]-buffer)

    random_goal_pos = np.random.uniform(-10.0, 10.0)
    random_goal_vel = np.random.uniform(limits['velocity'][0]+buffer, limits['velocity'][1]-buffer)
    random_goal_acc = np.random.uniform(limits['acceleration'][0]+buffer, limits['acceleration'][1]-buffer)

    start = [random_start_pos, random_start_vel, random_start_acc]
    goal = [random_goal_pos, random_goal_vel, random_goal_acc]
    
    return start, goal

def visualize_ruckig(traj, pred_y, t, itr, odefunc=None, show_plots=False):
    """ 
    Visualize the Ruckig trajectory and the predicted trajectory from the NODE.
    
    traj: (T, 5) numpy array with columns [time, pos, vel, acc, jerk]
    pred_y: (T, 1, 3) tensor with predicted [pos, vel, acc] from the NODE for a single trajectory
    t: (T) tensor with time points
    """
    
    if not args.viz:
        return

    # Convert tensors to numpy
    # Shape: (T, 1, 4) -> (T, 4)
    t_np = t.cpu().numpy()
    true_np = traj[:, 1:4]
    pred_np = pred_y.squeeze(1).cpu().detach().numpy()
    pred_y_squeezed = pred_y.squeeze(1) if pred_y.dim() == 3 else pred_y
    
    # print("pred_y_squeezed:", pred_y_squeezed.shape, "t.shape:", t.shape)

    # Create a figure with 4 vertical subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Titles and data mapping
    titles = ['Position', 'Velocity', 'Acceleration', 'Jerk']

    # Extracting Jerk
    true_jerk_np = traj[:, 4]
    
    if odefunc is not None:
        with torch.no_grad():
            # Get predicted jerk from the Controller's network
            pred_jerk_np = TrajectoryNODE.predict_jerk(pred_y_squeezed, multiplier=10.0).cpu().numpy()
    else:
        pred_jerk_np = np.zeros((len(t_np), 1))  # If no odefunc provided, just plot zeros for predicted jerk
    

    for i in range(4):
        ax = axes[i]
        ax.cla() # Clear current axis
        
        # Plotting
        if i < 3:
            # Plot Position (0), Velocity (1), Acceleration (2)
            ax.plot(t_np, true_np[:, i], 'g-', label='True' if i == 0 else "")
            ax.plot(t_np, pred_np[:, i], 'b--', label='Pred' if i == 0 else "")
        else:
            # Plot Jerk (3) - The control input you calculated in 'forward'
            ax.plot(t_np, true_jerk_np, 'g-', alpha=0.3, label='Ref Jerk')
            ax.plot(t_np, pred_jerk_np, 'b--', label='Neural Jerk')
        
        # Formatting
        ax.set_ylabel(titles[i], fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper right')
        if i == 3:
            ax.set_xlabel('Time (t)')

    plt.suptitle(f'Iteration: {itr:03d}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    # Save plot
    plt.savefig('png-test/traj_{:03d}.png'.format(itr))

    if show_plots:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.close()

def get_batch(true_y, t, batch_length=None):
    # s is a randomly sampled time along the length of the gt traj
    num_timesteps = len(t)
    
    batch_length = batch_length if batch_length is not None else args.batch_time
    
    window_size = min(batch_length, num_timesteps)
    
    max_start = max(0, num_timesteps - window_size)
    
    if max_start > 0:
        # Sample a single starting point 's' for the entire batch to keep time aligned
        s_val = np.random.choice(np.arange(max_start, dtype=np.int64))
        s = torch.ones(args.batch_size, dtype=torch.int64) * s_val
    else:
        s_val = 0
        s = torch.zeros(args.batch_size, dtype=torch.int64)
    
    # D - dimension of state (pos, vel, acc)
    # M - batch size (number of trajectory segments)
    # T - number of time points in each trajectory segment (window_size)
    
    batch_y0 = true_y[s].to(torch.float32)  # (M, D)
    
    # batch_t = t[:window_size].to(torch.float32)  # (T)
    batch_t = t[s_val : s_val + window_size].to(torch.float32)  # (T)
    
    batch_y = torch.stack([true_y[s + i] for i in range(window_size)], dim=0).to(torch.float32)  # (T, M, D)

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def thresholded_tanh(x, threshold=0.5):
    """
    Applies a thresholded hyperbolic tangent function to the input tensor.
    
    Args:
        x (Tensor): The input tensor.
        threshold (float/Tensor): The width of the zero-segment in the middle.
        
    Returns:
        Tensor: The activated tensor with a dead-zone of [-threshold, threshold].
    """
    # 1. Create a mask where values outside [-threshold, threshold] are True (1)
    mask = (x > threshold) | (x < -threshold)
    
    # 2. Shift the inputs toward zero so they start smoothly right outside the threshold
    #    (If x is positive, subtract threshold; if negative, add threshold)
    shifted_x = torch.where(x > 0, x - threshold, x + threshold)
    
    # 3. Apply tanh and zero out the dead-zone
    return torch.tanh(shifted_x) * mask.to(x.dtype)



class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.beta = 2.0 # Sharpness factor

    def forward(self, t, x_aug):
        # x_aug contains both the current state and the goal state: [pos, vel, acc, final_pos, final_vel, final_acc]
        pos_err = x_aug[:, 0:1]
        vel_err = x_aug[:, 1:2]
        acc_err = x_aug[:, 2:3]
        pos_goal = x_aug[:, 3:4]
        vel_goal = x_aug[:, 4:5]
        acc_goal = x_aug[:, 5:6]
        
        jerk = self.predict_jerk(x_aug, multiplier=10.0)
        # print("jerk:", jerk)
        djerk_goal = torch.zeros_like(acc_goal).to(torch.float32)
        
        return torch.cat([vel_err, acc_err, jerk, vel_goal, acc_goal, djerk_goal], dim=-1)   
    
    def predict_jerk(self, y, multiplier=1.0):
        ode_output = self.net(y)
        jerk_normalized = thresholded_tanh(self.beta * ode_output)
        return jerk_normalized * multiplier  

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def plot_loss(loss_values):
    plt.figure(figsize=(10, 6))
    
    plt.plot(loss_values, label='Training Loss', color='#1f77b4', linewidth=2)
    
    plt.title('Model Loss Over Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Set adjoint method for ODE integration
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    # Set device
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    
    # Create directories for saving plots and models if they don't exist
    if args.viz:
        makedirs('png-test') 
    if args.save:
        makedirs('models')
    
    # Controller
    TrajectoryNODE = Controller().to(device)
    
    # Load model
    if args.load_model:
        if args.load_model.exists():
            print(f"Loading checkpoint from {args.load_model}...")
            checkpoint = torch.load(args.load_model, map_location=device)
            TrajectoryNODE.load_state_dict(checkpoint['state_dict'])
            print("Model loaded successfully!")
    else:
        print(f"Warning: No model found at {args.load_model}. Starting from scratch.")
    
    # grab limits from config
    config = load_config(args.ruckig_config)
    limits = config['constraints']
    buffer = 2.8
    
    # hyperparameters 
    optimizer = optim.Adam(TrajectoryNODE.parameters(), lr=5e-4)
    iters = args.niters
    loss_tracking = []
    loss_meter = RunningAverageMeter(0.97)
    end = time.time()
    
    
    for itr in range(1, iters + 1):
        print("iter:", itr)
        optimizer.zero_grad()
        
        # generate random start and goal states within limits, with some buffer to avoid infeasible trajectories
        start, goal = random_start_goal_pair(limits, buffer)
        # start, goal = config['system']['initial_state'], config['system']['goal_state']
                
        # Get GT Ruckig trajectory
        traj, t = get_ruckig_traj(start=start, goal=goal, plot_trajectory=False)
        true_y = torch.tensor(traj[:, 1:5], device=device).to(torch.float32) # (T, 4) where columns are [pos, vel, acc, jerk]
        t = torch.from_numpy(t).to(device).to(torch.float32)
        
        # Get batch of trajectory segments
        batch_y0, batch_t, batch_y = get_batch(true_y, t)  
        batch_y0_state = batch_y0[:, :3]  # Extract position, velocity, and acceleration from the initial states
        batch_y0_jerk = batch_y0[:, 3:4] # Extract jerk from the initial states
        
        # append goal state of batch to each initial state in the batch
        batch_y0_aug = (torch.cat([batch_y0_state, batch_y[-1, :, :3]], dim=-1))  # (M, 6) where columns are [pos, vel, acc, final_pos, final_vel, final_acc]
        
        
        pred_y = odeint(TrajectoryNODE, batch_y0_aug, batch_t).to(device)
        # print("pred_y shape:", pred_y.shape)  # should be (T, M, 6)
        
        pred_jerk = TrajectoryNODE.predict_jerk(pred_y, multiplier=10.0)
        # print("pred_jerk:", pred_jerk)
        
        # weighted loss
        pos_err_weight = 1.0
        vel_err_weight = 10.0
        acc_err_weight = 30.0
        jerk_err_weight = 140.0
        
        trajectory_loss =  pos_err_weight * torch.mean((pred_y[:, :, 0] - batch_y[:, :, 0])**2) + \
                vel_err_weight * torch.mean((pred_y[:, :, 1] - batch_y[:, :, 1])**2) + \
                acc_err_weight * torch.mean((pred_y[:, :, 2] - batch_y[:, :, 2])**2) + \
                jerk_err_weight * torch.mean((pred_jerk - batch_y[:, :, 3:4])**2)
                
        # Terminal loss (the error at the very last time step, index -1)
        final_pos_err = 100 * torch.mean((pred_y[-1, :, 0] - batch_y[-1, :, 0])**2)
        final_vel_err = 100 * torch.mean((pred_y[-1, :, 1] - batch_y[-1, :, 1])**2)
        final_acc_err = 100 * torch.mean((pred_y[-1, :, 2] - batch_y[-1, :, 2])**2)

        terminal_loss = 10.0 * (final_pos_err + final_vel_err + final_acc_err)

        # Combined loss
        loss = trajectory_loss + terminal_loss
    
        print("loss:", loss.item())
        
        torch.nn.utils.clip_grad_norm_(TrajectoryNODE.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item())
        
        # test/validation loss and visualization
        if itr % args.test_freq == 0:
            
            # reshape start and goal to tensors and concatenate for input to TrajectoryNODE
            start_tensor = torch.tensor(start, device=device).to(torch.float32)
            goal_tensor = torch.tensor(goal, device=device).to(torch.float32)
            start_goal_aug = torch.cat([start_tensor, goal_tensor], dim=-1).unsqueeze(0)  # shape (1, 6)
            
            with torch.no_grad():
                pred_y_full = odeint(TrajectoryNODE, start_goal_aug, t).to(device)
                # print("pred_y_full shape:", pred_y_full.shape)  # should be (T, 1, 6)
                # print("true_y shape:", true_y.shape)  # should be (T, 4)
                # print("pred_jerk_full shape:", pred_jerk_full.shape)  # should be (T, 1, 1)
                
                pred_jerk_full = TrajectoryNODE.predict_jerk(pred_y_full, multiplier=10.0)
                
                trajectory_loss_full =  pos_err_weight * torch.mean((pred_y_full[:, 0, 0] - true_y[:, 0])**2) + \
                                        vel_err_weight * torch.mean((pred_y_full[:, 0, 1] - true_y[:, 1])**2) + \
                                        acc_err_weight * torch.mean((pred_y_full[:, 0, 2] - true_y[:, 2])**2) + \
                                        jerk_err_weight * torch.mean((pred_jerk_full[:, 0, 0] - true_y[:, 3])**2)
                            
                # Terminal loss (the error at the very last time step, index -1)
                final_pos_err_full = 100 * torch.mean((pred_y_full[-1, 0, 0] - true_y[-1, 0])**2)
                final_vel_err_full = 100 * torch.mean((pred_y_full[-1, 0, 1] - true_y[-1, 1])**2)
                final_acc_err_full = 100 * torch.mean((pred_y_full[-1, 0, 2] - true_y[-1, 2])**2)

                terminal_loss_full = 10.0 * (final_pos_err_full + final_vel_err_full + final_acc_err_full)

                # Combined loss
                loss_full = trajectory_loss_full + terminal_loss_full
                
                print('Iter {:04d} | Batch Loss {:.6f} | Full Trajectory Loss {:.6f}'.format(itr, loss_meter.avg, loss_full.item()))
                
                visualize_ruckig(traj, pred_y_full, t, itr, TrajectoryNODE, show_plots=False)
                
                loss_tracking.append(loss_full.item())
                
                if args.save:
                    torch.save(
                    {
                        "state_dict": TrajectoryNODE.state_dict(),
                    },
                    f"{args.save}/model.pt",
                    )
                  
        # Curriculum learning  
        if loss_meter.avg < 3000 and args.batch_time < 1500: # Threshold for "mastery"
            args.batch_time += 10
            # TrajectoryNODE.beta = min(15.0, TrajectoryNODE.beta + 2.0)
            print("Mastered current length, increasing batch time to {}".format(args.batch_time))
        
        end = time.time()
        
    plot_loss(loss_tracking)