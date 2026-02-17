import os
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import ruckig
from ruckig_generator import ruckig_generator, parse_args, load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "triple_integrator.json"

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
parser.add_argument("--save", type=str, default="model")
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
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

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

def visualize_ruckig(true_y, pred_y, t, itr, show_plots=False):

    print("true_y", true_y.shape)
    print("pred_y", pred_y.shape)

    if not args.viz:
        return

    # Create a figure with 4 vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Titles and data mapping
    titles = ['Position', 'Velocity', 'Acceleration', 'Jerk']
    colors_true = 'g-'  # Green solid for Ground Truth
    colors_pred = 'b--' # Blue dashed for Prediction
    
    # Convert tensors to numpy
    # Shape: (T, 1, 4) -> (T, 4)
    t_np = t.cpu().numpy()
    true_np = true_y.squeeze().cpu().detach().numpy()
    pred_np = pred_y.squeeze().cpu().detach().numpy()

    print("true_np", true_np.shape)
    print("pred_np", pred_np.shape)

    for i in range(3):
        ax = axes[i]
        ax.cla() # Clear current axis
        
        # Plotting
        ax.plot(t_np, true_np[:, i], colors_true, label='True' if i == 0 else "")
        ax.plot(t_np, pred_np[:, i], colors_pred, label='Pred' if i == 0 else "")
        
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

def visualize_jerk(true_jerk, pred_jerk, t, itr, show_plots=False):
    if not args.viz:
        return

    # Create a single plot
    plt.figure(figsize=(10, 5))
    
    # Convert tensors to numpy
    # Assuming shape (134, 1) or (134,)
    t_np = t.cpu().numpy()
    true_np = true_jerk.squeeze().cpu().detach().numpy()
    pred_np = pred_jerk.squeeze().cpu().detach().numpy()

    # Plotting
    plt.plot(t_np, true_np, 'g-', label='True Jerk')
    plt.plot(t_np, pred_np, 'b--', label='Predicted Jerk')

    # Formatting
    plt.title(f'Jerk Profile - Iteration: {itr:03d}', fontsize=14)
    plt.xlabel('Time (t)')
    plt.ylabel('Jerk ($m/s^3$)', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.tight_layout()

    # Save plot
    plt.savefig('png/jerk_{:03d}.png'.format(itr))

    if show_plots:
        plt.draw()
        plt.pause(0.001)

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)


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

    # func = ODEFunc().to(device)
    func = Controller().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        print("iter:", itr)
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        print("batch", batch_y.shape)
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
                print("test_y0", test_y0.shape)
                pred_y = odeint(func, test_y0, t).to(device)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize_ruckig(true_y, pred_y, t, ii)

                torch.save(
                {
                    "state_dict": func.state_dict(),
                },
                f"{args.save}/model.pt",
                )

                ii += 1

        end = time.time()
