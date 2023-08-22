import numpy as np
import torch
import casadi as cs

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
        n_hidden_units
    )
    
    # Load test data, iterate over it and compare model implementations
    n_tests = 100
    jp.LoadDataForTest(batch_size=1)
    for k, (length, state_input, fps_vel, ends_vel) in zip(range(n_tests), jp.testDataLoader):
        state_input_Yu = jp.relativeStateRepresentationTorch(state_input)
        J_Yu = jp.model_J(state_input_Yu.cuda()).cpu().detach().numpy()
        
        state_input_cs = cs.DM(state_input.numpy().ravel())
        J_own = model_own.compute_length_invariant_jacobian(
            state_input_cs, rbf_centers, rbf_sigmas, lin_A
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
        n_hidden_units
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
        v_fps_own = model_own(state_input_cs, length_cs, rbf_centers, rbf_sigmas, lin_A) @ ends_vel_cs
        
        p_cs = cs.vertcat(length_cs, cs.vec(rbf_centers), rbf_sigmas, cs.vec(lin_A))
        v_fps_own2 = fps_vel_fcn(state_input_cs, ends_vel_cs, p_cs)

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


    n_feature_points = jp.model_J.nFPs
    n_hidden_units = jp.model_J.numHidden
    model_own = casadi_dlo_model.JacobianNetwork(
        n_feature_points,
        n_hidden_units
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



if __name__ == '__main__':
    test_feature_points_velocity_predictions()