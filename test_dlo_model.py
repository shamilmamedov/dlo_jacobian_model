import numpy as np
import torch

from RBF import Net_J, JacobianPredictor 
import casadi_jacobian_model


def test_JacobainNetwork():
    jp = JacobianPredictor()
    jp.LoadModelWeights()
    rbf_centers = jp.model_J.fc1.centres.cpu().detach().numpy()
    rbf_sigmas = jp.model_J.fc1.sigmas.cpu().detach().numpy()
    lin_A = jp.model_J.fc2.weight.cpu().detach().numpy()


    n_feature_points = jp.model_J.nFPs
    n_hidden_units = jp.model_J.numHidden
    model_own = casadi_jacobian_model.JacobianNetwork(
        n_feature_points,
        n_hidden_units
    )

    jp.LoadDataForTest(batch_size=1)
    for (length, state_input, fps_vel, ends_vel) in jp.testDataLoader:
        state_input = jp.relativeStateRepresentationTorch(state_input)
        state_input_np = np.array(state_input)

        J_Yu = jp.model_J(state_input.cuda()).cpu().detach().numpy()
        J_own = model_own(state_input_np, rbf_centers, rbf_sigmas, lin_A)

        
        np.testing.assert_array_almost_equal(
            J_Yu.ravel(),
            np.array(J_own).ravel()
        )
        print('AAA')


if __name__ == '__main__':
    test_JacobainNetwork()