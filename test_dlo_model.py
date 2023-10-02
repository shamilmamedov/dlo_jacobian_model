import numpy as np
import torch
import casadi as cs
import time

from RBF import JacobianPredictor 
import casadi_dlo_model


def test_JacobainNetwork():
    # Instantiate Yu's jacobian predictor and load weights
    jp = JacobianPredictor()
    jp.LoadModelWeights()
    rbf_centers = cs.DM(jp.model_J.fc1.centres.cpu().detach().numpy())
    rbf_sigmas = cs.DM(jp.model_J.fc1.sigmas.cpu().detach().numpy())
    lin_A = cs.DM(jp.model_J.fc2.weight.cpu().detach().numpy())

    # Instantiate casadi implementation
    n_feature_points = jp.model_J.nFPs
    n_hidden_units = jp.model_J.numHidden
    model_own = casadi_dlo_model.JacobianNetwork(
        n_feature_points,
        n_hidden_units,
        rbf_centers,
        rbf_sigmas,
        lin_A
    )
    
    # Load test data, iterate over it and compare model implementations
    n_tests = 100
    jp.LoadDataForTest(batch_size=1)
    for k, (length, state_input, fps_vel, ends_vel) in zip(range(n_tests), jp.testDataLoader):
        state_input_Yu = jp.relativeStateRepresentationTorch(state_input)
        J_Yu = jp.model_J(state_input_Yu.cuda()).cpu().detach().numpy()
        
        state_input_cs = cs.DM(state_input.numpy().ravel())
        J_own = model_own.compute_length_invariant_jacobian(
            state_input_cs
        )

        np.testing.assert_array_almost_equal(
            J_Yu.squeeze(),
            np.array(J_own)
        )


def test_feature_points_velocity_predictions():
    # Instantiate Yu's jacobian predictor and load weights
    jp = JacobianPredictor()
    jp.LoadModelWeights()
    rbf_centers = jp.model_J.fc1.centres.cpu().detach().numpy()
    rbf_sigmas = jp.model_J.fc1.sigmas.cpu().detach().numpy()
    lin_A = jp.model_J.fc2.weight.cpu().detach().numpy()

    # Instantiate casadi implementation
    n_feature_points = jp.model_J.nFPs
    n_hidden_units = jp.model_J.numHidden
    model_own = casadi_dlo_model.JacobianNetwork(
        n_feature_points,
        n_hidden_units,
        cs.DM(rbf_centers),
        cs.DM(rbf_sigmas),
        cs.DM(lin_A)
    )
    fps_vel_fcn = model_own.get_feature_points_velocity_fcn()

    # Load test data, iterate over it and compare model implementations
    n_tests = 100
    jp.LoadDataForTest(batch_size=1)
    for k, (length, state_input, fps_vel, ends_vel) in zip(range(n_tests), jp.testDataLoader):
        fps_pos = state_input[:, 0:3*n_feature_points]
        ends_pose = state_input[:, 3*n_feature_points:3*n_feature_points +14]
        v_fps_Yu = jp.predFPsVelocities(length, fps_pos, ends_pose, ends_vel)

        length_cs = float(length)
        ends_vel_cs = cs.DM(ends_vel.numpy().ravel())
        state_input_cs = cs.DM(state_input.numpy().ravel())
        v_fps_own = model_own(state_input_cs, length_cs) @ ends_vel_cs
        v_fps_own2 = fps_vel_fcn(state_input_cs, ends_vel_cs, length_cs)

        np.testing.assert_array_almost_equal(
            v_fps_Yu.ravel(),
            np.array(v_fps_own).ravel()
        )

        np.testing.assert_array_almost_equal(
            v_fps_Yu.ravel(),
            np.array(v_fps_own2).ravel()
        )


def test_reshaping_jacobian(verbose: bool = False):
    n_fps = 2
    n_u = 4

    x = torch.arange(1, n_fps*n_u*3 + 1).reshape(1,-1)
    if verbose: print(f'Original vector\n{x}')
    x = torch.reshape(x, (-1, n_fps, n_u, 3))
    if verbose: print(f'First transformation\n{x}')
    x = x.transpose(2, 3)
    if verbose: print(f'Second transformation\n{x}')
    x = x.reshape(-1, 3 * n_fps, n_u)
    if verbose: print(f'Thrid transformation\n{x}')
    
    c = cs.DM(np.arange(1, n_fps*n_u*3 + 1))
    if verbose: print(f'\n\nOriginal casadi vector\n{c}')
    c = c.reshape((-1,n_fps))
    if verbose: print(f'First transformation\n{c}')
    c_cols = cs.horzsplit_n(c, n_fps)
    c = cs.vertcat(
        *[col.reshape((-1,n_u)) for col in c_cols]
    )
    if verbose: print(f'Second transformation\n{c}')

    np.testing.assert_array_almost_equal(
        x.detach().numpy().ravel(),
        np.array(c).ravel()
    )


def test_relative_position_computation():
    jp = JacobianPredictor()
    jp.LoadModelWeights()
    rbf_centers = jp.model_J.fc1.centres.cpu().detach().numpy()
    rbf_sigmas = jp.model_J.fc1.sigmas.cpu().detach().numpy()
    lin_A = jp.model_J.fc2.weight.cpu().detach().numpy()


    n_feature_points = jp.model_J.nFPs
    n_hidden_units = jp.model_J.numHidden
    model_own = casadi_dlo_model.JacobianNetwork(
        n_feature_points,
        n_hidden_units,
        cs.DM(rbf_centers),
        cs.DM(rbf_sigmas),
        cs.DM(lin_A)
    )

    n_tests = 100
    jp.LoadDataForTest(batch_size=1)
    for k, (length, state_input, fps_vel, ends_vel) in zip(range(n_tests), jp.testDataLoader):
        state_input_Yu = jp.relativeStateRepresentationTorch(state_input)
        state_input_own = model_own._compute_relative_positions(
            cs.DM(state_input.numpy().ravel())
        )
        
        np.testing.assert_array_almost_equal(
            state_input_Yu.numpy().ravel(),
            np.array(state_input_own).ravel()
        )


# A unit test that tests the pose prediction of the robot's end-effectors
# and compares it against calcNextEndsPose of the JacobianPredictor
def test_end_effectors_pose_prediction():
    q = np.random.uniform(-1, 1, size=(4,1))
    q = q / np.linalg.norm(q)
    p = np.random.uniform(-1, 1, size=(3,1))
    pose = np.vstack((p,q))
    vel = np.random.uniform(-1, 1, size=(6,1))
    delta_t = 0.1

    jp = JacobianPredictor()
    
    current_ends_pose = np.vstack((pose, pose))
    ends_vel = np.vstack((vel, vel))

    next_ends_pose = jp.calcNextEndsPose(current_ends_pose.T, ends_vel.T, delta_t)
    print(f'Yu\'s next pos {next_ends_pose[0,:3]}')
    print(f'Yu\'s next quat {next_ends_pose[0,3:7]}')

    # Casadi implementation
    rhs = casadi_dlo_model.end_effector_pose_dynamics(cs.DM(pose), cs.DM(vel))
    next_pose = pose + delta_t * rhs
    print(f'Own next pos {next_pose[:3].T}')
    print(f'Own next quat {next_pose[3:7].T}')
    print(f'Own quat norm {cs.norm_2(next_pose[3:7])}')


def _load_model_parameters():
    # Instantiate Yu's jacobian predictor and load weights
    jp = JacobianPredictor()
    jp.LoadModelWeights()
    rbf_centers = cs.DM(jp.model_J.fc1.centres.cpu().detach().numpy())
    rbf_sigmas = cs.DM(jp.model_J.fc1.sigmas.cpu().detach().numpy())
    lin_A = cs.DM(jp.model_J.fc2.weight.cpu().detach().numpy())

    # Instantiate casadi implementation
    n_feature_points = jp.model_J.nFPs
    n_hidden_units = jp.model_J.numHidden

    return {'n_feature_points': n_feature_points,
            'n_hidden_units': n_hidden_units,
            'rbf_centers': rbf_centers,
            'rbf_sigmas': rbf_sigmas,
            'lin_A': lin_A}


def test_DualArmDLOModel():
    params = _load_model_parameters()
    dlo_model = casadi_dlo_model.JacobianNetwork(**params)
    dlo_length = 0.5

    setup_model = casadi_dlo_model.DualArmDLOModel(dlo_model, dlo_length)

    # Test dual arm dynamics
    poses, vels, poses_dot = setup_model._get_dual_arm_dynamics_expr()
    assert poses.shape == (14, 1)
    assert vels.shape == (12, 1)
    assert poses_dot.shape == (14, 1)

    # Test setup dynamics
    z, u, z_dot = setup_model._get_setup_dynamics_expr()
    assert z.shape == (14+3*params['n_feature_points'], 1)
    assert u.shape == (12, 1)
    assert z_dot.shape == (14+3*params['n_feature_points'], 1)
    


if __name__ == '__main__':
    test_DualArmDLOModel()
    # test_JacobainNetwork()
    # test_end_effectors_pose_prediction()
    # test_feature_points_velocity_predictions()