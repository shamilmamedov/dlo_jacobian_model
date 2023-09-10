import numpy as np
import matplotlib.pyplot as plt

import visualization


if __name__ == '__main__':
    n_traj = 8
    data_dir = '../../dataset/sim_data/'
    traj_name = f'state_{n_traj}.npy'
    filepath = data_dir + traj_name
    
    data = np.load(filepath)

    dt = 0.1
    cable_length = data[0,0]
    X_particles = data[1:,1:31]
    dX_particles = data[1:,45:75]
    X_particles_goal = data[1:, 87:117]
    print(f'Cable length: {cable_length} m')

    Z = np.hstack((X_particles[:,3:-3], dX_particles[:,3:-3]))
    U = np.hstack((X_particles[:,:3], dX_particles[:,:3],
                   X_particles[:,-3:], dX_particles[:,-3:]))

    ns2keep = 100
    X_lee = data[:ns2keep, 31:34]
    Phi_lee = data[:ns2keep, 34:38]
    X_ree = data[:ns2keep, 38:41]
    Phi_ree = data[:ns2keep, 41:45]

    dX_lee = data[:ns2keep, 75:78]
    dPhi_lee = data[:ns2keep, 78:81]
    dX_ree = data[:ns2keep, 81:84]
    dPhi_ree = data[:ns2keep, 84:87]

    # Plot change of orientation
    t = dt*np.arange(X_lee.shape[0])

    _, axs = plt.subplots(4,1, sharex=True)
    axs.reshape(-1)
    for ax, o1, o2 in zip(axs, X_lee.T, X_ree.T):
        ax.plot(t, o1)
        ax.plot(t, o2)
        ax.grid(alpha=0.1)
    plt.tight_layout()
    plt.show()

    X = X_particles[:100,:]
    X_goal = X_particles_goal[:100,:]
    visualization.visualize_mass_spring_system(X, X_goal, dt, n_replays=5)