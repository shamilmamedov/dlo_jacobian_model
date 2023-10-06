import numpy as np
import torch
import casadi as cs
import time

from RBF import JacobianPredictor 
import dlo_jacobian_model.casadi_dlo_model as casadi_dlo_model


def measure_avg_JacobainNetwork_evaluation():
    # Instantiate Yu's jacobian predictor and load weights
    model_params = casadi_dlo_model.load_model_parameters()
    model = casadi_dlo_model.JacobianNetwork(
        **model_params
    )
    
    # Load test data, iterate over it and compare model implementations
    avg_time = []
    n_tests = 25
    jp = JacobianPredictor()
    jp.LoadDataForTest(batch_size=1)
    for k, (length, state_input, fps_vel, ends_vel) in zip(range(n_tests), jp.testDataLoader):
        state_input_cs = cs.DM(state_input.numpy().ravel())
        start = time.time()
        for k in range(1000):
            J_own = model.compute_length_invariant_jacobian(
                state_input_cs
            )
        end = time.time()
        avg_time.append((end-start)/1000)
        # print(f'Jacobian evaluation took {(end-start)/1000} seconds')

    print(f'Average Jacobian evaluation time: {1000*np.mean(avg_time):.3f}ms')
    print(f'Max Jacobian evaluation time: {1000*np.max(avg_time):.3f}ms\n')


def measure_avg_setup_model_evaluation():
    # Instantiate Yu's jacobian predictor and load weights
    model_params = casadi_dlo_model.load_model_parameters()
    dlo_model = casadi_dlo_model.JacobianNetwork(
        **model_params
    )
    dlo_length = 0.5
    setup_model = casadi_dlo_model.DualArmDLOModel(dlo_model, dlo_length)

    # Load test data, iterate over it and compare model implementations
    avg_time = []
    n_tests = 25
    jp = JacobianPredictor()
    jp.LoadDataForTest(batch_size=1)
    for k, (length, state_input, fps_vel, ends_vel) in zip(range(n_tests), jp.testDataLoader):
        state_input_cs = cs.DM(state_input.numpy().ravel())
        ends_vel_cs = cs.DM(ends_vel.numpy().ravel())
        start = time.time()
        for k in range(1000):
            J_own = setup_model(state_input_cs, ends_vel_cs)
        end = time.time()
        avg_time.append((end-start)/1000)
        # print(f'Setup dynamics evaluation took {(end-start)/1000} seconds')

    print(f'Average setup dynamics evaluation time: {1000*np.mean(avg_time):.3f}ms')
    print(f'Max setup dynamics evaluation time: {1000*np.max(avg_time):.3f}ms\n')


def measure_avg_linearized_setup__dynamics_evaluation():
    model_params = casadi_dlo_model.load_model_parameters()
    dlo_model = casadi_dlo_model.JacobianNetwork(
        **model_params
    )
    dlo_length = 0.5
    setup_model = casadi_dlo_model.DualArmDLOModel(dlo_model, dlo_length)
    A_fcn, B_fcn = setup_model._get_linearized_setup_dynamics_fcns(jit=True)

    # Load test data, iterate over it and compare model implementations
    avg_time_A = []
    avg_time_B = []
    n_tests = 5
    jp = JacobianPredictor()
    jp.LoadDataForTest(batch_size=1)
    for k, (length, state_input, fps_vel, ends_vel) in zip(range(n_tests), jp.testDataLoader):
        state_input_cs = cs.DM(state_input.numpy().ravel())
        ends_vel_cs = cs.DM(ends_vel.numpy().ravel())
        start = time.time()
        for k in range(1000):
            J_own = A_fcn(state_input_cs, ends_vel_cs)
        end = time.time()
        avg_time_A.append((end-start)/1000)
        print(f'A matrix evalution time took {1000*avg_time_A[-1]:.3f} ms')

        start = time.time()
        for k in range(1000):
            J_own = B_fcn(state_input_cs, ends_vel_cs)
        end = time.time()
        avg_time_B.append((end-start)/1000)
        print(f'B matrix evalution time took {1000*avg_time_B[-1]:.3f} ms')

    print(f'Average A matrix evaluation time: {1000*np.mean(avg_time_A):.3f}ms')
    print(f'Max A matrix evaluation time: {1000*np.max(avg_time_A):.3f}ms')
    print(f'Average B matrix evaluation time: {1000*np.mean(avg_time_B):.3f}ms')
    print(f'Max B matrix evaluation time: {1000*np.max(avg_time_B):.3f}ms')


if __name__ == '__main__':
    measure_avg_JacobainNetwork_evaluation()
    measure_avg_setup_model_evaluation()
    measure_avg_linearized_setup__dynamics_evaluation()