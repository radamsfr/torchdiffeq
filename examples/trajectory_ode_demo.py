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
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--ruckig_config', type=Path, default=DEFAULT_CONFIG_PATH)
parser.add_argument('--load_model', nargs='?', type=Path, const=DEFAULT_SAVED_MODEL, default=None)
parser.add_argument("--save", nargs='?', type=str, const="models", default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


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
    traj, t = get_ruckig_traj(False)

    # traj.shape (t*dt, 1, d) d = (time, pos, vel, acc, jerk)
    # print("traj:", traj)

    true_y = torch.from_numpy(traj[:,1:4]).to(device).to(torch.float32)
    true_y0 = true_y[0, :].to(device).to(torch.float32)

    # print("true_y0", true_y0)

    t = torch.from_numpy(t).to(device)


def get_batch():
    # s is a randomly sampled time along the length of the gt traj
    s = torch.from_numpy(np.random.choice(np.arange(len(t) - args.batch_time, dtype=np.int64), args.batch_size, replace=False))

    batch_y0 = true_y[s].to(torch.float32)  # (M, D)
    batch_t = t[:args.batch_time].to(torch.float32)  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0).to(torch.float32)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')

if args.save:
    makedirs('model')


def visualize(true_y, pred_y, odefunc, itr):

    # visualize the gt (ruckig) trajectory and neuralODE generated trajectory

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)

def visualize_ruckig(true_y, pred_y, t, itr, odefunc=None, show_plots=False):
    if not args.viz:
        return

    # Convert tensors to numpy
    # Shape: (T, 1, 4) -> (T, 4)
    t_np = t.cpu().numpy()
    true_np = traj[:, 1:4]
    pred_np = pred_y.squeeze().cpu().detach().numpy()
    pred_y_squeezed = pred_y.squeeze(1) if pred_y.dim() == 3 else pred_y

    # Create a figure with 4 vertical subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Titles and data mapping
    titles = ['Position', 'Velocity', 'Acceleration', 'Jerk']

    # Extracting Jerk
    if odefunc is not None:
        with torch.no_grad():
            # Get predicted jerk from the Controller's network
            pred_jerk_np = odefunc.net(pred_y_squeezed).cpu().numpy()
            true_jerk_np = traj[:, 4]
    else:
        pred_jerk_np = np.zeros((len(t_np), 1))
        true_jerk_np = np.zeros((len(t_np), 1))

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
    plt.savefig('png/traj_{:03d}.png'.format(itr))

    if show_plots:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.close()


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


if __name__ == '__main__':

    ii = 0
    
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
    
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        print("iter:", itr)
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        # test/validation loss
        if itr % args.test_freq == 0:
            with torch.no_grad():
                test_y0 = true_y0.unsqueeze(0)
                pred_y = odeint(func, test_y0, t).to(device).squeeze(1)
                
                # print("pred_y", pred_y.shape)
                # print("true_y", true_y.shape)

                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize_ruckig(true_y, pred_y, t, ii, func, False)

                if args.save:
                    torch.save(
                    {
                        "state_dict": func.state_dict(),
                    },
                    f"{args.save}/model.pt",
                    )

                ii += 1

        end = time.time()
