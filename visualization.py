import matplotlib.pyplot as plt
from matplotlib import animation
import pinocchio as pin
import hppfcl as fcl
import numpy as np
from panda3d_viewer import Viewer, ViewerConfig
from pinocchio.visualize import Panda3dVisualizer, MeshcatVisualizer
import time


# Panda3D configs
PANDA3D_CONFIG = ViewerConfig()
PANDA3D_CONFIG.enable_antialiasing(True, multisamples=4)
PANDA3D_CONFIG.enable_shadow(False)
PANDA3D_CONFIG.show_axes(False)
PANDA3D_CONFIG.show_grid(True)
PANDA3D_CONFIG.show_floor(True)
PANDA3D_CONFIG.enable_spotlight(True)
PANDA3D_CONFIG.enable_hdr(True)
PANDA3D_CONFIG.set_window_size(1125, 1000)


def visualize_mass_spring_system(q, q_goal, dt, n_replays, visualizer='panda3d'):
    add_goal_positions = False
    if q_goal is not None:
        add_goal_positions = True

    n_particles = q.shape[1] // 3
    model, geom_model = create_mass_spring_pinocchio_model(n_particles, add_goal_positions)
    vis_model = geom_model

    if visualizer == 'panda3d':
        viz = Panda3dVisualizer(model, collision_model=geom_model, visual_model=geom_model)
        viewer = Viewer(config=PANDA3D_CONFIG)
        viewer.set_background_color(((255, 255, 255)))
        viewer.reset_camera((4, 0, 1), look_at=(0,0,0))
        viz.initViewer(viewer=viewer)
        viz.loadViewerModel(group_name=f'{model.name}')
    if visualizer == 'meshcat':
        viz = MeshcatVisualizer(model, geom_model, vis_model)
        viz.initViewer()
        viz.loadViewerModel()
    
    if q_goal is not None:
        q = np.hstack((q, q_goal))

    viz.displayVisuals(True)
    for _ in range(n_replays):
        viz.display(q[0, :])
        time.sleep(2)
        if visualizer == 'panda3d':
            viz.play(q[1:, :], dt)
        if visualizer == 'meshcat':
            viz.play(q[1:, :], dt)
        time.sleep(1)
    viz.viewer.stop()


def create_mass_spring_pinocchio_model(n_particles: int = 5, add_goal_positions: bool = False):
    particle_joint = pin.JointModelTranslation()

    # Instantiate a model
    model = pin.Model()
    model.name = 'Mass-Spring-System'
    geom_model = pin.GeometryModel()
    parent_id = 0

    # Geometrical paramters of particles
    particle_radius = 0.01
    particle_shape = fcl.Sphere(particle_radius)
    c_particle_shape = fcl.Box(0.015, 0.015, 0.015)

    # Defailt joint and body placements
    joint_placement = pin.SE3.Identity()
    body_placement = pin.SE3.Identity()
    mesh_color = np.array([0.6, 0.6, 0.6, 1.])

    for k in range(n_particles):
        # Add joint to a model
        joint_name = f'joint_{k}'
        joint_id = model.addJoint(
            parent_id,
            particle_joint,
            joint_placement,
            joint_name
        )

        # Inertia parameters of a particle
        body_inertia = pin.Inertia.Zero()
        model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())

        # Define geometry for visualization
        if k == 0 or k == n_particles - 1:
            shape = c_particle_shape
        else:
            shape = particle_shape
        geom_name = f'particle_{k}'
        shape_placement = body_placement
        geom_obj = pin.GeometryObject(geom_name, joint_id, shape, shape_placement)
        geom_obj.meshColor = np.array([0.6, 0.6, 0.6, 1.])
        geom_model.addGeometryObject(geom_obj)

    # if add_goal_positions is true then copy code from above and create joints and geometyry for goal positions
    if add_goal_positions:
        for k in range(n_particles):
            # Add joint to a model
            joint_name = f'goal_joint_{k}'
            joint_id = model.addJoint(
                parent_id,
                particle_joint,
                joint_placement,
                joint_name
            )

            # Inertia parameters of a particle
            body_inertia = pin.Inertia.Zero()
            model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())

            # Define geometry for visualization
            if k == 0 or k == n_particles - 1:
                shape = c_particle_shape
            else:
                shape = particle_shape
            geom_name = f'goal_particle_{k}'
            shape_placement = body_placement
            geom_obj = pin.GeometryObject(geom_name, joint_id, shape, shape_placement)
            geom_obj.meshColor = np.array([0.961, 0.294, 0.082, 1.])
            geom_model.addGeometryObject(geom_obj)
    

    return model, geom_model


if __name__ == "__main__":
    l = 0.3
    n = 10
    l_i = l/n

    xx = np.arange(0., l+l_i, l_i)[:,None]
    xy = np.zeros_like(xx)
    xz = np.ones_like(xx)
    X0 = np.hstack((xx, xy, xz)).reshape(-1,1)
    
    xz_goal = np.ones_like(xx) * 0.5
    X0_goal = np.hstack((xx, xy, xz_goal)).reshape(-1,1)
    
    dt = 0.1

    q = np.tile(X0.T, (50,1))
    q_goal = np.tile(X0_goal.T, (50,1))
    visualize_mass_spring_system(q, q_goal, dt, 5)