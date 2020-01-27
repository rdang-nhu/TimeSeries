import argparse

import matplotlib

import attack_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import json
import numpy as np

def plot_batch(original_mu ,original_sigma,
               perturbed_output_mu, perturbed_output_sigma,
               best_c, best_perturbation ,best_distance, labels,
               targets, params):

    batch_size = original_mu.shape[0]

    # sample_metrics = utils.get_metrics(original_mu,
    #                                   labels,
    #                                   params.test_predict_start,
    #                                   samples,
    #                                   relative=params.relative_metrics)


    all_samples = np.arange(batch_size)
    if batch_size < 12:  # make sure there are enough unique samples to choose bottom 90 from
        random_sample = np.random.choice(all_samples, size=10, replace=True)
    else:
        random_sample = np.random.choice(all_samples, size=10, replace=False)

    label_plot = labels[random_sample]
    original_mu_chosen = original_mu[random_sample]
    original_sigma_chosen = original_sigma[random_sample]
    plot_target_double = targets["double"][random_sample]
    plot_target_zero = targets["zero"][random_sample]

    # plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}

    x = np.arange(params["test_window"])

    nrows = 10
    ncols = 1

    for tolerance in range(perturbed_output_mu["double"].shape[0]):
        f = plt.figure(figsize=(8, 42), constrained_layout=True)
        ax = f.subplots(nrows, ncols)

        for k in range(nrows):

            ax[k].plot(x[params["predict_start"]:],
                       original_mu_chosen[k], color='b')
            ax[k].fill_between(x[params["predict_start"]:],
                               original_mu_chosen[k] - \
                               2 * original_sigma_chosen[k],
                               original_mu_chosen[k] + \
                               2 * original_sigma_chosen[k], color='blue',
                               alpha=0.2)

            double_mu_chosen = perturbed_output_mu["double"][tolerance][random_sample]
            zero_mu_chosen = perturbed_output_mu["zero"][tolerance][random_sample]

            ax[k].plot(x[params["predict_start"]:],
                       double_mu_chosen[k], color='black')

            ax[k].plot(x[params["predict_start"]:],
                       zero_mu_chosen[k], color='brown')

            double_pert = ( 1 +best_perturbation["double"][tolerance][: ,random_sample])
            zero_pert = (1 + best_perturbation["zero"][tolerance][: ,random_sample])

            print(double_pert[k])

            ax[k].plot(x[:params["predict_start"]], label_plot[k, :params["predict_start"]] *
                       double_pert[:params["predict_start"] ,k], color='y')
            ax[k].plot(x[:params["predict_start"]:], label_plot[k, :params["predict_start"]] *
                       zero_pert[:params["predict_start"] ,k], color='purple')

            ax[k].axhline(plot_target_double[k], color='orange', linestyle='dashed')
            ax[k].axhline(plot_target_zero[k], color='orange', linestyle='dashed')

            ax[k].plot(x, label_plot[k, :], color='r')
            ax[k].axvline(params["predict_start"], color='g', linestyle='dashed')

        # ax[k].set_title(plot_metrics_str, fontsize=10)

        name = 'plot_tolerance_'+str(params["tolerance"][tolerance])+'.png'
        f.savefig(os.path.join(params["output_folder"],name))
        plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_folder', help='Output folder for plots')
    parser.add_argument('--debug', action="store_true", help='Debug mode')

    # Load the parameters
    args = parser.parse_args()

    # Load data from output folder
    folder = os.path.join("attack_logs",args.output_folder)
    loader = attack_utils.H5pySaver(folder)

    original_mu = loader.get_from_file('original_mu')
    original_sigma = loader.get_from_file('original_sigma')
    best_c = loader.get_dict_from_file('best_c')
    best_perturbation = loader.get_dict_from_file('best_perturbation')
    best_distance = loader.get_dict_from_file('best_distance')
    perturbed_output_mu = loader.get_dict_from_file('perturbed_output_mu')
    perturbed_output_sigma = loader.get_dict_from_file('perturbed_output_sigma')
    targets = loader.get_dict_from_file('targets')
    labels = loader.get_from_file('labels')

    with open(os.path.join(folder,"params.txt")) as json_file:
        params = json.load(json_file)

    # Call plotting function
    plot_batch(original_mu,
               original_sigma,
               perturbed_output_mu,
               perturbed_output_sigma,
               best_c,
               best_perturbation,
               best_distance,
               labels,
               targets,
               params)
