import numpy as np
import torch
import casadi as cs
import time

from RBF import JacobianPredictor 
import casadi_dlo_model


def measure_avg_JacobainNetwork_evaluation():
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
    avg_time = []
    n_tests = 25
    jp.LoadDataForTest(batch_size=1)
    for k, (length, state_input, fps_vel, ends_vel) in zip(range(n_tests), jp.testDataLoader):
        state_input_cs = cs.DM(state_input.numpy().ravel())
        start = time.time()
        for k in range(1000):
            J_own = model_own.compute_length_invariant_jacobian(
                state_input_cs
            )
        end = time.time()
        avg_time.append((end-start)/1000)
        print(f'Casadi implementation took {(end-start)/1000} seconds')

    print(f'Average evaluation time: {1000*np.mean(avg_time):.3f}ms')
    print(f'Max evaluation time: {1000*np.max(avg_time):.3f}ms')


def measure_avg_setup_model_evaluation():
    # Instantiate Yu's jacobian predictor and load weights
    jp = JacobianPredictor()
    jp.LoadModelWeights()
    rbf_centers = cs.DM(jp.model_J.fc1.centres.cpu().detach().numpy())
    rbf_sigmas = cs.DM(jp.model_J.fc1.sigmas.cpu().detach().numpy())
    lin_A = cs.DM(jp.model_J.fc2.weight.cpu().detach().numpy())

    # Instantiate casadi implementation
    n_feature_points = jp.model_J.nFPs
    n_hidden_units = jp.model_J.numHidden
    dlo_model = casadi_dlo_model.JacobianNetwork(
        n_feature_points,
        n_hidden_units,
        rbf_centers,
        rbf_sigmas,
        lin_A
    )
    dlo_length = 0.5
    setup_model = casadi_dlo_model.DualArmDLOModel(dlo_model, dlo_length)

    # Load test data, iterate over it and compare model implementations
    avg_time = []
    n_tests = 25
    jp.LoadDataForTest(batch_size=1)
    for k, (length, state_input, fps_vel, ends_vel) in zip(range(n_tests), jp.testDataLoader):
        state_input_cs = cs.DM(state_input.numpy().ravel())
        ends_vel_cs = cs.DM(ends_vel.numpy().ravel())
        start = time.time()
        for k in range(1000):
            J_own = setup_model(state_input_cs, ends_vel_cs)
        end = time.time()
        avg_time.append((end-start)/1000)
        print(f'Casadi implementation took {(end-start)/1000} seconds')

    print(f'Average evaluation time: {1000*np.mean(avg_time):.3f}ms')
    print(f'Max evaluation time: {1000*np.max(avg_time):.3f}ms')


if __name__ == '__main__':
    # measure_avg_JacobainNetwork_evaluation()
    measure_avg_setup_model_evaluation()