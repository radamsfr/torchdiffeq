import os
import argparse
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch
import torch.nn as nn
import torch.optim as optim

import ruckig
from ruckig_generator import ruckig_generator, parse_args, load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "triple_integrator.json"
DEFAULT_SAVED_MODEL = Path(__file__).resolve().parents[1] / "models" / "model.pt"

parser = argparse.ArgumentParser('TrajectoryODE')
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--ruckig_config', type=Path, default=DEFAULT_CONFIG_PATH)
parser.add_argument('--load_model', nargs='?', type=Path, const=DEFAULT_SAVED_MODEL, default=None)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_ruckig_traj(plot_trajectory=False):

    config = load_config(args.ruckig_config)
    
    rg = ruckig_generator(config)

    # Extract start and goal, vel and acc limits, dt from config
    start, goal = rg.extract_states(config)
    limits = rg.extract_limits(config)
    dt = rg.extract_dt(config)

    print("Ruckig Start state:", start)
    print("Ruckig Goal state:", goal)
    print("Ruckig Limits:", limits)
    print("Ruckig dt:", dt)

    # Build Ruckig input
    inp = rg.build_ruckig_input()

    # Run Ruckig
    traj, t = rg.run_ruckig(inp)
    
    # print("traj:\n", traj)
    print("Ruckig shape:", traj.shape)

    # Plot results
    if plot_trajectory:
        rg.plot_trajectory(t, traj)

    print("traj:", traj.shape, type(traj))
    print("t:", t.shape)

    return traj, t


with torch.no_grad():
    traj, t = get_ruckig_traj()

    # traj.shape (t*dt, 1, d) d = (time, pos, vel, acc, jerk)
    # print("traj:", traj)

    # true_y = torch.from_numpy(traj[:,1:4]).to(device).to(torch.float32)
    true_y0 = torch.from_numpy(traj[0, 1:4]).to(device).to(torch.float32)

    # print("true_y0", true_y0)

    t = torch.from_numpy(t).to(device)
    
    
class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # # Start near u = 0
        # nn.init.zeros_(self.net[-1].weight)
        # nn.init.zeros_(self.net[-1].bias)

        # self.A = torch.tensor([
        #     [0., 1., 0.],
        #     [0., 0., 1.],
        #     [0., 0., 0.]
        # ]).to(device)

        # self.B = torch.tensor([
        #     [0.],
        #     [0.],
        #     [1.]
        # ]).to(device)

    def forward(self, t, x):
            pos = x[:, 0:1]
            acc = x[:, 2:3]
            vel = x[:, 1:2]

            # print("pos:", pos.shape)
            # print("vel:", vel.shape)
            # print("acc:", acc.shape)
            
            # Predict ONLY the jerk (control input)
            jerk = self.net(x)

            # print("jerk:", jerk)
            
            # Return the derivative of the state: [v, a, j]
            return torch.cat([vel, acc, jerk], dim=-1)
        
        
def visualize_ruckig_dual(true_y, pred_y, t, odefunc=None, show_plots=False):

    # Convert tensors to numpy
    # Shape: (T, 1, 4) -> (T, 4)
    t_np = t.cpu().numpy()
    true_np = traj[:, 1:4]
    pred_np = pred_y.squeeze().cpu().detach().numpy()
    pred_y_squeezed = pred_y.squeeze(1) if pred_y.dim() == 3 else pred_y

    line_thickness = 2.0
    label_font_size = 16
    tick_font_size = 12
    label_spacing = 0
    
    # Create a figure with 4 vertical subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    
    # Titles and data mapping
    titles = ['Position', 'Velocity', 'Acceleration', 'Jerk']

    # Extracting Jerk
    if odefunc is not None:
        with torch.no_grad():
            # Get predicted jerk from the Controller's network
            pred_jerk_np = odefunc.net(pred_y_squeezed).cpu().numpy()
            true_jerk_np = traj[:, 4]
            
            # print("true_j", true_jerk_np.shape)
            # print("pred_j", pred_jerk_np.shape) 
    else:
        pred_jerk_np = np.zeros((len(t_np), 1))
        true_jerk_np = np.zeros((len(t_np), 1))

    for i in range(4):
        # ax = axes[i]
        # ax.cla() # Clear current axis
        
        # ax.set_ylim(-10.5, 10.5)
        
        y_lims = [-10.5, 10.5]
        
        # --- LEFT COLUMN: NEURAL ODE ---
        ax_l = axes[i, 0]
        ax_l.set_ylim(y_lims)
        data_l = pred_np[:, i] if i < 3 else pred_jerk_np
        ax_l.plot(t_np, data_l, 'b--', linewidth=line_thickness)
        ax_l.set_ylabel(f"Neural {titles[i]}", fontweight='semibold', fontsize=label_font_size, labelpad=label_spacing)
        ax_l.grid(True, linestyle=':', alpha=0.6)

        for label in ax_l.get_xticklabels() + ax_l.get_yticklabels():
            label.set_fontweight('semibold')
            label.set_fontsize(tick_font_size)

        # --- RIGHT COLUMN: GT RUCKIG ---
        ax_r = axes[i, 1]
        ax_r.set_ylim(y_lims)
        data_r = true_np[:, i] if i < 3 else true_jerk_np
        ax_r.plot(t_np, data_r, 'g-', linewidth=line_thickness)
        ax_r.set_ylabel(f"GT {titles[i]}", fontweight='semibold', fontsize=label_font_size, labelpad=label_spacing)
        ax_r.grid(True, linestyle=':', alpha=0.6)

        for label in ax_r.get_xticklabels() + ax_r.get_yticklabels():
            label.set_fontweight('semibold')
            label.set_fontsize(tick_font_size)
            
        if i == 0:
            ax_l.set_title("TrajectoryODE", fontsize=20, pad=10)
            ax_r.set_title("Ruckig (Ground Truth)", fontsize=20, pad=10)
        if i == 3:
            ax_l.set_xlabel("Time (t)", fontsize=16)
            ax_r.set_xlabel("Time (t)", fontsize=16)


    # plt.suptitle(f'Predicted Trajectory', fontsize=16)        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(hspace=0.3, wspace=0.15)
    
    # Save plot
    plt.savefig(f"results/traj_({true_y0[0]},{true_y0[1]},{true_y0[2]}).png")

    if show_plots:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.close()
        
def visualize_ruckig_overlay(true_y, pred_y, t, odefunc=None, show_plots=False):
    # Convert tensors to numpy
    t_np = t.cpu().numpy()
    # true_y is usually [T, 3] or similar, pred_y is [T, 1, 3]
    true_np = true_y.cpu().detach().numpy() 
    pred_y_squeezed = pred_y.squeeze(1) if pred_y.dim() == 3 else pred_y
    pred_np = pred_y_squeezed.cpu().detach().numpy()

    line_thickness = 2.0
    label_font_size = 26
    legend_font_size = 19
    tick_font_size = 19
    label_spacing = 5 # Increased slightly for overlayed look
    
    # Back to 4 rows, 1 column
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    
    titles = ['Position', 'Velocity', 'Acceleration', 'Jerk']

    # Extracting Jerk
    if odefunc is not None:
        with torch.no_grad():
            pred_jerk_np = odefunc.net(pred_y_squeezed).cpu().numpy()
            # Assuming you have global access to 'traj' or it's passed in. 
            # If 'traj' is not available here, you'll need to pass it as an argument.
            true_jerk_np = traj[:, 4] 
    else:
        pred_jerk_np = np.zeros((len(t_np), 1))
        true_jerk_np = np.zeros((len(t_np), 1))

    for i in range(4):
        ax = axes[i]
        
        # Determine current data
        if i < 3:
            y_true = true_np[:, i]
            y_pred = pred_np[:, i]
        else:
            y_true = true_jerk_np
            y_pred = pred_jerk_np

        # --- PLOT BOTH ON SAME AXIS ---
        ax.plot(t_np, y_true, 'g-', linewidth=line_thickness, label='GT Ruckig')
        ax.plot(t_np, y_pred, 'b--', linewidth=line_thickness, label='Neural ODE')
        
        # Ticks
        ymin, ymax = [-10, 10]
        ax.set_yticks([ymin, 0, ymax])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        
        # Styling
        ax.set_ylabel(titles[i], fontweight='semibold', fontsize=label_font_size, labelpad=label_spacing)
        ax.grid(True, linestyle=':')

        # Semibold ticks
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('semibold')
            label.set_fontsize(tick_font_size)
            
        # Add legend to the first plot only to keep it clean, or all
        if i == 0:
            ax.legend(loc='upper right', fontsize=legend_font_size, frameon=False)
        
        if i == 3:
            ax.set_xlabel("Time (t)", fontsize=label_font_size, fontweight='semibold')

    # plt.suptitle(f"Trajectory Overlay:\nPred vs Ground Truth", fontsize=30, fontweight='semibold')
    
    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
    
    # Create results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/overlay_traj.png")

    if show_plots:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.close()

if __name__ == "__main__":
    
    makedirs('results')
    
    func = Controller().to(device)
    
    # Load model
    if args.load_model:
        if args.load_model.exists():
            print(f"Loading checkpoint from {args.load_model}...")
            checkpoint = torch.load(args.load_model, map_location=device)
            func.load_state_dict(checkpoint['state_dict'])
            print("Model loaded successfully!")
    else:
        print(f"Warning: No model found at {args.load_model}. Starting from scratch.")
            
    with torch.no_grad():
                test_y0 = true_y0.unsqueeze(0)
                true_y = torch.from_numpy(traj[:,1:4]).to(device).to(torch.float32)
                pred_y = odeint(func, test_y0, t).to(device).squeeze(1)
                
                # print("pred_y", pred_y.shape)
                # print("true_y", true_y.shape)

                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Total Loss {:.6f}'.format(loss.item()))
                visualize_ruckig_overlay(true_y, pred_y, t, odefunc=func, show_plots=False)