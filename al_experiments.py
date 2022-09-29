import numpy as np
import torch
import jsonargparse
import pickle
import time
import pandas as pd

from utils.data_handler import MyDataLoader
from simulators.simulator import oracle_simulator
from utils.active_learning import apply_active_learning_strategy
from utils.transformations import transform
from utils.gp_utils import get_model, get_likelihood, get_loss, compute_loss

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Set directory name (path)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def do_active_learning(args):
    if args.simulator is not None:
        oracle = oracle_simulator(args)
    else:
        oracle = None
    data = MyDataLoader(args=args, oracle=oracle)
    data.get_initial_data(seed=args.seed)
    data.compute_true_mean_and_stddev()

    # Active learning scheme
    nmll_losses, nmll_losses_valid = [], []
    rmse_losses_valid, rrse_losses_valid, mae_losses_valid, rmse_std_losses_valid = [], [], [], []
    df = pd.DataFrame(columns=['lengthscale0', 'noise0', 'iteration', 'data_size'])
    print(f"[AL_toy_simulator.py] Starting active learning for: {args.metamodel_name}")
    tic = time.time()
    for i in range(args.active_learning_steps):
        data.transform()
        data.make_dataloader()

        # Fit model
        model = get_model(args, data, likelihood=get_likelihood(args))
        nmll_loss, fit_losses, outs = model.fit(data.gp_train_loader, args)
        nmll_losses.append(nmll_loss)

        # Validate with: marginal log likelihood (MLL)
        predict_output = model.predict(data.gp_test_loader)
        predictions_mll = predict_output['predictions']

        losses_valid = compute_loss(args, data.gp_test_loader, predictions_mll, lst_metrics=['mll'],
                              mll=get_loss(args, model, num_data=data.train_trans.y.numel()))
        nmll_losses_valid.append(losses_valid['nmll'])

        # Make predictions on the search space
        data.get_candidate_points(seed=None)
        pred_x_trans, _, _ = transform(torch.Tensor(data.candidate_points), data.x_mu, data.x_sigma,
                                       method=args.transformation_x)
        predict_output_querying = model.predict(dataloader=(pred_x_trans, None))
        predict_output = predict_output_querying

        # Transform pred_x, mean and standard deviation back
        pred_x = transform(pred_x_trans, data.x_mu, data.x_sigma, method=args.transformation_x, inverse=True)
        pred_mean = transform(predict_output['mean'], data.y_mu, data.y_sigma, method=args.transformation_y, inverse=True)
        pred_std = transform(predict_output['stddev'], 0, data.y_sigma, method=args.transformation_y, inverse=True)
        pred_std_f = transform(predict_output['stddev_f'], 0, data.y_sigma, method=args.transformation_y, inverse=True)

        # Compute loses
        losses = compute_loss(args, dataloader=(None, data.true_mean_cp), predictions=pred_mean,
                              lst_metrics=['rmse', 'rrse'])
        rmse_losses_valid.append(losses['rmse'].tolist())
        rrse_losses_valid.append(losses['rrse'].tolist())
        losses = compute_loss(args, dataloader=(None, data.true_std_cp), predictions=pred_std,
                              lst_metrics=['rmse'])
        rmse_std_losses_valid.append(losses['rmse'].tolist())

        # Compute sampling strategy (evaluate acqusition function)
        sample_strategy_output, selection_array, new_points, data = \
            apply_active_learning_strategy(args, model, data, predict_output_querying, i)

    # Save things to file
    p_dct = {}
    p_dct['data'] = data
    p_dct['train_x'] = data.train.x
    p_dct['train_y'] = data.train.y
    p_dct['test_x'] = data.test.x
    p_dct['test_y'] = data.test.y
    p_dct['pred_x'] = pred_x
    p_dct['mean'] = pred_mean
    p_dct['pred_std'] = pred_std
    p_dct['nmll_losses'] = nmll_losses
    p_dct['nmll_losses_valid'] = nmll_losses_valid
    p_dct['rmse_losses_valid'] = rmse_losses_valid
    p_dct['rrse_losses_valid'] = rrse_losses_valid
    p_dct['mae_losses_valid'] = mae_losses_valid
    p_dct['rmse_std_losses_valid'] = rmse_std_losses_valid
    p_dct['new_points'] = args.k_samples * args.repeat_sampling
    p_dct['running_time'] = time.time() - tic
    if args.model_type in ['fbgp_mcmc']:
        p_dct['mcmc_samples'] = model.mcmc_samples
        df.reset_index(drop=True)
        df['seed'] = args.seed
        p_dct['df_mcmc_samples'] = df
        p_dct['model_state_dict'] = None
    else:
        p_dct['model_state_dict'] = [m.state_dict() for m in model] if isinstance(model, list) else model.state_dict()
    p_dct['args'] = args
    p_dct['fit_losses'] = fit_losses

    # Convert from tensor to list
    for k in p_dct.keys():
        if torch.is_tensor(p_dct[k]):
            p_dct[k] = p_dct[k].tolist()

    if not os.path.exists(f'{args.folder}'):
        os.mkdir(f'{args.folder}')
    with open(f'{args.folder}{args.metamodel_name}.pkl', 'wb') as fp:
        fp.write(pickle.dumps(p_dct, protocol=4))
    print(f"Wrote results to {args.folder}{args.metamodel_name}. The experiment took {p_dct['running_time']} sec.")


if __name__ == '__main__':
    parser = jsonargparse.ArgumentParser(default_config_files=[dname + '/config_fbgp.yaml'])
    parser.add_argument('--cfg', action=jsonargparse.ActionConfigFile)
    parser.add_argument('--al_type', type=str)
    parser.add_argument('--seed', type=int)
    # Data
    parser.add_argument('--outputs', type=int)
    parser.add_argument('--initial_samples', type=int)
    parser.add_argument('--space_filling_design', type=str)
    parser.add_argument('--test_samples', type=int)
    parser.add_argument('--transformation_x', type=str)
    parser.add_argument('--transformation_y', type=str)
    parser.add_argument('--simulator', type=str)
    parser.add_argument('--noise_level', type=float)
    # Model
    parser.add_argument('--model_type', type=str, help="Which model to use")
    parser.add_argument('--covar_type', type=str, help="Choice of kernel (only ExactGP)")
    parser.add_argument('--covar_ard', type=bool, help="Apply Automatic Relevance Determinantion in kernel")
    parser.add_argument('--covar_prior', type=str, help="Prior on hyperparameters in kernel")
    parser.add_argument('--mean_type', type=str, help="Choice of mean function (only ExactGP)")
    parser.add_argument('--noise_prior', type=bool, help="Add prior on the noise")
    parser.add_argument('--initial_lr', type=float, help="Initial learning rate")
    parser.add_argument('--milestones', help="List with milestones for learning rate schedules")
    parser.add_argument('--n_epochs', type=int, help="Number of epochs for fitting model")
    parser.add_argument('--num_chains', type=int, help="Number of chains in MCMC")
    parser.add_argument('--num_samples', type=int, help="Number of samples in MCMC")
    parser.add_argument('--warmup_steps', type=int, help="Number of warmsteps in MCMC")
    parser.add_argument('--predict_mcmc', type=str, help="Method to use then predicting with FBGP")
    # Active learning
    parser.add_argument('--selection_criteria', type=str, help="Active learning acquisition function")
    parser.add_argument('--active_learning_steps', type=int, help="Number of active learning steps")
    parser.add_argument('--k_samples', type=int, help="Number of distinct samples")
    parser.add_argument('--repeat_sampling', type=int, help="Number of repeat samples")
    # Outputs
    parser.add_argument('--folder', type=str)
    parser.add_argument('--metamodel_name', type=str)

    args = parser.parse_args()

    # Run the active learning
    do_active_learning(args)
