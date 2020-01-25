import torch.nn as nn
import torch

import logging
import argparse
import os

import model.net as net
import utils


class AttackLoss(nn.Module):

    def __init__(self, params,c):
        super(AttackLoss, self).__init__()
        self.c = c
        self.device = params.device

    # perturbation has shape (nSteps,)
    # output has shape (nSteps,batch_size,output_dim)
    # for the moment, target has shape (batch_size,output_dim)
    def forward(self, perturbation, output, target):

        output = output[:,-1]

        loss_function = nn.MSELoss(reduction="none")
        distance_per_sample = loss_function(output, target)

        distance = distance_per_sample.sum(0)

        zero = torch.zeros(perturbation.shape).to(self.device)
        norm_per_sample = loss_function(perturbation, zero).sum(0)

        norm = norm_per_sample.sum(0)

        loss_per_sample = norm_per_sample + self.c * distance_per_sample
        loss = norm + self.c * distance

        return norm_per_sample,distance_per_sample,loss_per_sample,norm,distance,loss


def forward_model(model,data,id_batch,v_batch,hidden,cell,params):

    for t in range(params.test_predict_start):
        # if z_t is missing, replace it by output mu from the last time step
        zero_index = (data[t, :, 0] == 0)
        if t > 0 and torch.sum(zero_index) > 0:
            data[t, zero_index, 0] = mu[zero_index]

        mu, sigma, hidden, cell = model(data[t].unsqueeze(0), id_batch, hidden, cell)

    samples, sample_mu, sample_sigma = model.test(data,
                                                  v_batch,
                                                  id_batch,
                                                  hidden,
                                                  cell,
                                                  sampling=True,
                                                  n_samples=params.batch_size)
    return samples,sample_mu,sample_sigma

def set_params():

    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)

    params = utils.Params(json_path)

    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.c = args.c
    params.n_iterations = args.n_iterations
    params.tolerance = args.tolerance
    params.batch_size = args.batch_size
    params.learning_rate = args.lr

    return params,model_dir,args,data_dir

def set_cuda(params,logger):
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

    return model