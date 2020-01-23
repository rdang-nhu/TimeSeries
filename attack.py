import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.net as net
from dataloader import *

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'

# Attack parameters
parser.add_argument('--c',nargs='+', type=float, default=[0.01, 0.1, 1, 10, 100],
                        help='list of c coefficients (see Carlini et al.)')
parser.add_argument('--n_iterations', type=int, default=1000,
                        help='Number of iterations for attack')
parser.add_argument('--tolerance', nargs='+',type=float, default=0.1,help='Max perturbation L2 norm')

parser.add_argument('--debug', action="store_true", help='Debug mode')

class Attack():
    '''Attack the model.
            Args:
                model: (torch.nn.Module) the Deep AR model
                loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
                test_loader: load test data and labels
                params: (Params) hyperparameters
                plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        '''

    def __init__(self,model, loss_fn, test_loader, params, plot_num):
        self.model = model
        self.loss_fn = loss_fn
        self.test_loader = test_loader
        self.params = params
        self.plot_num = plot_num

        # Set the model to eval mode
        self.model.eval()
        self.max_pert_len = len(params.tolerance)


    def attack_batch(self, data):

            shape = (self.max_pert_len,) + data.shape
            best_perturbation = {"buy": np.zeros(shape),
                                 "sell": np.zeros(shape)}

            c_shape = (self.max_pert_len, data.shape[1])
            best_c = {"buy": np.zeros(c_shape),
                      "sell": np.zeros(c_shape)}
            best_distance = {"buy": np.full(c_shape, np.inf),
                             "sell": np.full(c_shape, np.inf)}

            percentage = {}

            modes = ["buy", "sell"]

            for mode in modes:
                # Loop on values of c to find successful attack with minimum perturbation

                for i in range(len(self.params.c)):

                    c = self.params.c[i]

                    # Update the lines
                    attack_module = AttackModule(self.model, self.params, c, data)

                    target, mean_output = attack_module.generate_target()

                    optimizer = optim.RMSprop([attack_module.perturbation], lr=self.args.learning_rate)

                    # Iterate steps
                    for i in range(self.params.n_iterations):

                        if self.estimator == "ours":
                            self.attack_step_ours(attack_module, optimizer, i, target)
                        elif self.estimator == "naive":
                            self.attack_step_naive(attack_module, optimizer, i, target)
                        else:
                            raise Exception("No such estimator")

                    # Evaluate the attack
                    # Run full number of samples on perturbed input to obtain perturbed output
                    with torch.no_grad():
                        perturbed_output = attack_module(n_samples=self.args.samples)

                        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
                            attack_module.attack_loss(attack_module.perturbation, perturbed_output, target)

                        # Find
                        numpy_norm = np.sqrt(utils.convert_from_tensor(norm_per_sample))
                        numpy_distance = utils.convert_from_tensor(distance_per_sample)
                        numpy_perturbation = utils.convert_from_tensor(attack_module.perturbation.data)

                        for l in range(self.max_pert_len):
                            indexes_best_c = np.logical_and(numpy_norm <= self.args.max_pert[l],
                                                            numpy_distance < best_distance[mode][l])

                            best_perturbation[mode][l][:, indexes_best_c] = \
                                numpy_perturbation[:, indexes_best_c]
                            best_distance[mode][l, indexes_best_c] = \
                                numpy_distance[indexes_best_c]
                            best_c[mode][l, indexes_best_c] = c

                with torch.no_grad():
                    if self.target_type == "binary":
                        percentage[mode] = []
                        for l in range(self.max_pert_len):
                            # Check if 95% confidence interval is in "buy" or "sell"
                            attack_module.perturbation.data = \
                                torch.tensor(best_perturbation[mode][l],
                                             device=attack_module.model.device).float()
                            perturbed_output, sem = attack_module(n_samples=self.args.samples, std=True)

                            percentage[mode].append(self.compute_metrics(mode, perturbed_output, sem, mean_output))

            return best_c, best_perturbation, best_distance, percentage


    def attack(self):


        # For each test sample
        # Test_loader:
        # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
        # id_batch ([batch_size]): one integer denoting the time series id;
        # v ([batch_size, 2]): scaling factor for each window;
        # labels ([batch_size, train_window]): z_{1:T}.
        for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(self.test_loader)):

            print("Sample",i)
            print(test_batch.shape)

            self.attack_batch(test_batch)

            #

        # Average the performance across batches



    '''
    with torch.no_grad():
        plot_batch = np.random.randint(len(test_loader) - 1)

        summary_metric = {}
        raw_metrics = utils.init_metrics(sample=sample)

        
        # loop
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
            id_batch = id_batch.unsqueeze(0).to(params.device)
            v_batch = v.to(torch.float32).to(params.device)
            labels = labels.to(torch.float32).to(params.device)
            batch_size = test_batch.shape[1]
            input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device)  # scaled
            input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device)  # scaled
            hidden = model.init_hidden(batch_size)
            cell = model.init_cell(batch_size)

            for t in range(params.test_predict_start):
                # if z_t is missing, replace it by output mu from the last time step
                zero_index = (test_batch[t, :, 0] == 0)
                if t > 0 and torch.sum(zero_index) > 0:
                    test_batch[t, zero_index, 0] = mu[zero_index]

                mu, sigma, hidden, cell = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell)
                input_mu[:, t] = v_batch[:, 0] * mu + v_batch[:, 1]
                input_sigma[:, t] = v_batch[:, 0] * sigma

            if sample:
                samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell,
                                                              sampling=True)
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels,
                                                   params.test_predict_start, samples, relative=params.relative_metrics)
            else:
                sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell)
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels,
                                                   params.test_predict_start, relative=params.relative_metrics)

            if i == plot_batch:
                if sample:
                    sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, samples,
                                                       relative=params.relative_metrics)
                else:
                    sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start,
                                                       relative=params.relative_metrics)
                    # select 10 from samples with highest error and 10 from the rest
                top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // 10]  # hard coded to be 10
                chosen = set(top_10_nd_sample.tolist())
                all_samples = set(range(batch_size))
                not_chosen = np.asarray(list(all_samples - chosen))
                if batch_size < 100:  # make sure there are enough unique samples to choose top 10 from
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
                else:
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
                if batch_size < 12:  # make sure there are enough unique samples to choose bottom 90 from
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
                else:
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
                combined_sample = np.concatenate((random_sample_10, random_sample_90))

                label_plot = labels[combined_sample].data.cpu().numpy()
                predict_mu = sample_mu[combined_sample].data.cpu().numpy()
                predict_sigma = sample_sigma[combined_sample].data.cpu().numpy()
                plot_mu = np.concatenate((input_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
                plot_sigma = np.concatenate((input_sigma[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
                plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}
                plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, params.test_window,
                                   params.test_predict_start, plot_num, plot_metrics, sample)

        summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
        metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
        logger.info('- Full test metrics: ' + metrics_string)
    return summary_metric


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):
    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                           predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                           alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        # metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})

        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
                           f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()
    '''

if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))


    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.c = args.c
    params.n_iterations = args.n_iterations
    params.tolerance = args.tolerance

    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    attack = Attack(model, loss_fn, test_loader, params, -1,)
    test_metrics = attack.attack()
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    #utils.save_dict_to_json(test_metrics, save_path)