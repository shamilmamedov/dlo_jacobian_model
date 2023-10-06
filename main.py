import casadi as cs
import numpy as np

from RBF import JacobianPredictor 
import dlo_jacobian_model.casadi_dlo_model as casadi_dlo_model


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


def setup_model():
    """
    Example of calling the setup model that returns the time derivative of
    z = [x_dlo, p_l, q_l, p_r, q_r]
    """
    jac_model_params = _load_model_parameters()
    dlo_model = casadi_dlo_model.JacobianNetwork(**jac_model_params)
    dlo_length = 0.5

    setup_model = casadi_dlo_model.DualArmDLOModel(dlo_model, dlo_length)

    x = np.random.rand((setup_model.nx))
    u = np.random.rand((setup_model.nu))

    z_dot = setup_model(x, u)


def dlo_model():
    """
    Example of calling the dlo model that returns the jacobian of the DLO
    wrt to the feature points 
    """
    jac_model_params = _load_model_parameters()
    dlo_model = casadi_dlo_model.JacobianNetwork(**jac_model_params)
    dlo_length = 0.5

    x = np.random.rand((3*jac_model_params['n_feature_points'] + 14, 1))

    jac = dlo_model(x, dlo_length)
