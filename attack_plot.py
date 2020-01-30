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


    nrows = 3
    ncols = 3
    size = nrows*ncols

    all_samples = np.arange(batch_size)
    if batch_size < 12:  # make sure there are enough unique samples to choose bottom 90 from
        random_sample = np.random.choice(all_samples, size=size, replace=True)
    else:
        random_sample = np.random.choice(all_samples, size=size, replace=False)

    for mode in ["double","zero"]:

        label_plot = labels[random_sample]
        original_mu_chosen = original_mu[random_sample]
        original_sigma_chosen = original_sigma[random_sample]
        plot_target_double = targets["double"][random_sample]

        # plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}

        x = np.arange(params["test_window"])

        target_index = -7
        for tolerance in range(perturbed_output_mu["double"].shape[0]):
            f = plt.figure(figsize=(20,7), constrained_layout=True)
            ax = f.subplots(nrows, ncols)

            for k0 in range(nrows):

                for k1 in range(ncols):

                    k = nrows * k1 + k0


                    ax[k0][k1].plot(x[:target_index + 1],
                                    np.concatenate([label_plot[k,:params["predict_start"]],
                                    original_mu_chosen[k][:target_index+1]]),
                                    color='r',
                                    linewidth=4)

                    ax[k0][k1].fill_between(x[params["predict_start"]:target_index + 1],
                                            original_mu_chosen[k][:target_index + 1] - \
                                            original_sigma_chosen[k][:target_index + 1],
                                            original_mu_chosen[k][:target_index + 1] + \
                                            original_sigma_chosen[k][:target_index + 1], color='r',
                                            alpha=0.2)

                    # Plot original prediction
                    #ax[k0][k1].plot(x[params["predict_start"]:target_index+1],
                    #           original_mu_chosen[k][:target_index+1], color='b')

                    # Plot uncertainty

                    mu_chosen = perturbed_output_mu[mode][tolerance][random_sample]
                    sigma_chosen = perturbed_output_sigma[mode][tolerance][random_sample]

                    ax[k0][k1].fill_between(x[params["predict_start"]:target_index+1],
                                       mu_chosen[k][:target_index+1] - \
                                        sigma_chosen[k][:target_index+1],
                                       mu_chosen[k][:target_index+1] + \
                                        sigma_chosen[k][:target_index+1], color='blue',
                                       alpha=0.2)

                    pert = (1 + best_perturbation[mode][tolerance][1:, random_sample])

                    aux = label_plot[k, :params["predict_start"]] * pert[:params["predict_start"] ,k]

                    # Plot adversarial sample 1
                    ax[k0][k1].plot(x[:target_index+1],
                               np.concatenate([aux,mu_chosen[k][:target_index+1]]),
                                    color='blue',
                                    linewidth=2)


                    #ax[k0][k1].axhline(plot_target_double[k], color='orange', linestyle='dashed')


                    ax[k0][k1].axvline(params["predict_start"], color='g', linestyle='dashed')

                    ax[k0][k1].set_ylim(ymin=0)
                    ax[k0][k1].grid()

                    if k0 == 2 and k1 == 1:
                        ax[k0][k1].set_xlabel("Hour")

                    if k0 == 1 and k1 == 0:
                        ax[k0][k1].set_ylabel("Electricity consumption")



            # ax[k].set_title(plot_metrics_str, fontsize=10)
            str_tol = str(params["tolerance"][tolerance])
            #plt.suptitle("Tolerance "+str_tol)
            name = mode+'_plot_tolerance_'+str_tol+'.png'
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
