import argparse

import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import pandas as pd

import attack_plot
import attack_utils
import utils
import model.net as net
from attack_utils import AttackLoss
from dataloader import *

logger = logging.getLogger('DeepAR.Eval')

class AttackModule(nn.Module):

    def __init__(self, model, params, c, data,id_batch, v_batch, hidden, cell):

        super(AttackModule, self).__init__()

        self.model = model
        self.params = params
        self.c = c
        self.data = data
        self.id_batch = id_batch
        self.v_batch = v_batch
        self.hidden = hidden
        self.cell = cell
        self.n_inputs = data.shape[1]
        self.attack_loss = AttackLoss(params,c,v_batch)

        # Initialize perturbation
        self.perturbation = nn.Parameter(torch.zeros(self.data.shape[:2], device=self.params.device))

    def generate_target(self,labels,mode):

        if mode == "double":
            target = 2*labels
        elif mode == "zero":
            target = torch.zeros(labels.shape,device=self.params.device)
        else:
            raise Exception("No such mode")

        return target

    # Returns mean and std
    def forward(self):

        perturbed_data = torch.zeros(self.data.shape).to(self.params.device)
        perturbed_data[:,:,0] = self.data[:,:,0]  * (1 + self.perturbation)
        perturbed_data[:,:,1:] = self.data[:,:,1:]

        samples, sample_mu, sample_sigma = attack_utils.forward_model(model,
                                                                      perturbed_data,
                                                                      self.id_batch,
                                                                      self.v_batch,
                                                                      self.hidden,
                                                                      self.cell,
                                                                      self.params)

        return samples,sample_mu,sample_sigma

    # Not clear yet how to compute that
    def forward_naive(self):

        # Forward pass on all samples
        aux_estimate = torch.zeros(batch,device=self.model.device)
        for i in range(n_samples):
            log_prob = self.model.forward_log_prob()

            aux_estimate += outputs[i, :,self.args.steps-1].squeeze(1)*log_prob

        aux_estimate /= float(n_samples)
        aux_estimate = aux_estimate.sum(0)

        return sample_mu,aux_estimate


class Attack():

    def __init__(self,model, loss_fn, test_loader, params, plot_num):
        self.model = model
        self.loss_fn = loss_fn
        self.test_loader = test_loader
        self.params = params
        self.plot_num = plot_num

        # Set the model to eval mode
        # Replaced model.eval() to deal with
        # RuntimeError: cudnn RNN backward can only be called in training mode
        # (https: // github.com / pytorch / pytorch / issues / 10006)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0

            elif isinstance(module, nn.LSTM):
                module.dropout = 0

        print('model: ', model)

        self.max_pert_len = len(params.tolerance)

    def print(self,i,norm,distance,loss,batch):

        print(i,norm.detach().item()/batch,
              distance.detach().item()/batch,
              loss.detach().item()/batch)

    def project_perturbation(self,attack_module):

        aux = torch.tensor([-1.], device=self.params.device)

        attack_module.perturbation.data = torch.max(attack_module.perturbation.data, aux)


    def attack_step_ours(self, attack_module, optimizer, i, target):

        attack_module.zero_grad()

        # with torch.autograd.detect_anomaly():
        _,prediction,_ = attack_module()

        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
            attack_module.attack_loss(attack_module.perturbation, prediction, target)

        # Differentiate loss with respect to input
        loss.backward()

        if i % 100 == 0:
            print("Iteration", i)
            self.print(i,norm,distance,loss,norm_per_sample.shape[0])

        # Apply one step of optimizer
        optimizer.step()

        self.project_perturbation(attack_module)

    def attack_step_naive(self, attack_module, optimizer, i, target):

        attack_module.zero_grad()

        mean, aux_estimate = attack_module.forward_naive()

        aux_estimate.backward()

        # Compute the derivative of the loss with respect to the mean
        mean.requires_grad = True
        attack_module.perturbation.requires_grad = False
        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
            attack_module.attack_loss(attack_module.perturbation, mean, target)

        # This propagates the gradient to the mean
        loss.backward()

        # Compute grad of aux estimate

        # Multiply the two, and set it in perturbation
        attack_module.perturbation.grad *= mean.grad.unsqueeze(-1)

        # Compute the derivative of the loss with respect to the norm
        mean.requires_grad = False
        attack_module.perturbation.requires_grad = True
        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
            attack_module.attack_loss(attack_module.perturbation, mean, target)

        # This propagates the gradient to the norm
        loss.backward()


        # Apply one step of optimizer
        optimizer.step()

        self.project_perturbation(attack_module)

    def attack_batch(self, data, id_batch, v_batch, labels, hidden, cell, estimator):

            with torch.no_grad():
                _, original_mu, original_sigma = attack_utils.forward_model(model,
                                                                          data,
                                                                          id_batch,
                                                                          v_batch,
                                                                          hidden,
                                                                          cell,
                                                                          self.params)

            shape = (self.max_pert_len,) + data.shape[:2]

            best_perturbation = {"double": np.zeros(shape),
                                 "zero": np.zeros(shape)}

            c_shape = (self.max_pert_len, data.shape[1])
            best_c = {"double": np.zeros(c_shape),
                      "zero": np.zeros(c_shape)}
            best_distance = {"double": np.full(c_shape, np.inf),
                             "zero": np.full(c_shape, np.inf)}

            perturbed_output_mu = {}
            perturbed_output_sigma = {}

            modes = ["double", "zero"]
            targets = {}

            lines = []

            for mode in modes:

                perturbed_output_mu[mode] = {}
                perturbed_output_sigma[mode] = {}

                # Loop on values of c to find successful attack with minimum perturbation

                for i in range(len(self.params.c)):

                    c = self.params.c[i]
                    print("c",c)

                    # Update the lines
                    attack_module = AttackModule(self.model,
                                                 self.params,
                                                 c,
                                                 data,
                                                 id_batch,
                                                 v_batch,
                                                 hidden,
                                                 cell)

                    target = attack_module.generate_target(labels,mode)
                    targets[mode] = target

                    optimizer = optim.Adam([attack_module.perturbation], lr=self.params.learning_rate)

                    # Iterate steps
                    for i in range(self.params.n_iterations):


                        if estimator == "ours":
                            self.attack_step_ours(attack_module, optimizer, i, target)
                        elif estimator == "naive":
                            self.attack_step_naive(attack_module, optimizer, i, target)
                        else:
                            raise Exception("No such estimator")

                    # Evaluate the attack
                    # Run full number of samples on perturbed input to obtain perturbed output
                    with torch.no_grad():
                        _,perturbed_output,_ = attack_module()

                        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
                            attack_module.attack_loss(attack_module.perturbation, perturbed_output, target)

                        # Find
                        numpy_norm = np.sqrt(utils.convert_from_tensor(norm_per_sample))
                        numpy_distance = utils.convert_from_tensor(distance_per_sample)
                        numpy_perturbation = utils.convert_from_tensor(attack_module.perturbation.data)

                        #print("numpy perturbation",attack_module.perturbation.data[:,0])

                        print("Final", i)
                        self.print(i, norm, distance, loss, norm_per_sample.shape[0])

                        for l in range(self.max_pert_len):

                            indexes_best_c = np.logical_and(numpy_norm <= self.params.tolerance[l],
                                                            numpy_distance < best_distance[mode][l])

                            best_perturbation[mode][l][:, indexes_best_c] = \
                                numpy_perturbation[:, indexes_best_c]
                            best_distance[mode][l, indexes_best_c] = \
                                numpy_distance[indexes_best_c]
                            best_c[mode][l, indexes_best_c] = c

                        # Save norm and distance for c plot
                        mean_numpy_norm = np.mean(numpy_norm)
                        mean_distance = np.mean(np.sqrt(numpy_distance))

                        lines.append([mode,c,mean_numpy_norm,mean_distance])

                with torch.no_grad():

                    for l in range(self.max_pert_len):

                        attack_module.perturbation.data = \
                            torch.tensor(best_perturbation[mode][l],
                                         device=self.params.device).float()
                        _,aux1,aux2 = attack_module()

                        perturbed_output_mu[mode][l] = aux1
                        perturbed_output_sigma[mode][l] = aux2


            return original_mu,original_sigma,best_c, best_perturbation, \
                   best_distance, perturbed_output_mu, perturbed_output_sigma,\
                   targets,lines

    def attack(self):

        for estimator in ["ours"]:

            # Choose a batch on with to plot
            # plot_batch = np.random.randint(len(test_loader) - 1)
            plot_batch = 0

            print("params", self.params.predict_steps)

            # For each test sample
            # Test_loader:
            # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
            # id_batch ([batch_size]): one integer denoting the time series id;
            # v ([batch_size, 2]): scaling factor for each window;
            # labels ([batch_size, train_window]): z_{1:T}.
            for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(self.test_loader)):
                if i == plot_batch:

                    index = v[:,0] > 0
                    test_batch = test_batch[index]
                    id_batch = id_batch[index]
                    v = v[index]
                    labels = labels[index]

                    # Prepare batch data
                    test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
                    id_batch = id_batch.unsqueeze(0).to(params.device)
                    v_batch = v.to(torch.float32).to(params.device)
                    test_labels = labels.to(torch.float32).to(params.device)[:,-1]

                    #print(v_batch[:],torch.mean(labels,dim=0))

                    batch_size = test_batch.shape[1]
                    hidden = model.init_hidden(batch_size)
                    cell = model.init_cell(batch_size)

                    print("Sample", i)
                    #print("label",labels[0,:])

                    original_mu,original_sigma,best_c,best_perturbation,best_distance,\
                        perturbed_output_mu, perturbed_output_sigma,targets, lines = \
                        self.attack_batch(test_batch,id_batch,v_batch,test_labels,hidden,cell,estimator)

                    attack_plot.plot_batch(original_mu,
                                    original_sigma,
                                    perturbed_output_mu,
                                    perturbed_output_sigma,
                                    best_c,
                                    best_perturbation,
                                    best_distance,
                                    labels,
                                    targets,
                                    params)

                    df = pd.DataFrame(lines,columns=["mode","c","norm","distance"])
                    df.to_csv(os.path.join(params.output_folder,"results.csv"))


            # Average the performance across batches

            # Save results to dataframe

            # Plots some of adversarial samples

if __name__ == '__main__':

    params,model_dir,args,data_dir = attack_utils.set_params()

    model = attack_utils.set_cuda(params,logger)

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)


    loss_fn = net.loss_fn

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    attack = Attack(model, loss_fn, test_loader, params, -1,)
    test_metrics = attack.attack()
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    #utils.save_dict_to_json(test_metrics, save_path)