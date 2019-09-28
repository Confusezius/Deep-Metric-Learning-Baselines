# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


#################### LIBRARIES ########################
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn
import auxiliaries as aux
import datasets as data

import netlib as netlib
import losses as losses
import evaluate as eval

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

####### Main Parameter: Dataset to use for Training
parser.add_argument('--dataset',      default='cub200',   type=str, help='Dataset to use.')

### General Training Parameters
parser.add_argument('--lr',                default=0.00001,  type=float, help='Learning Rate for network parameters.')
parser.add_argument('--fc_lr_mul',         default=0,        type=float, help='OPTIONAL: Multiply the embedding layer learning rate by this value. If set to 0, the embedding layer shares the same learning rate.')
parser.add_argument('--n_epochs',          default=70,       type=int,   help='Number of training epochs.')
parser.add_argument('--kernels',           default=8,        type=int,   help='Number of workers for pytorch dataloader.')
parser.add_argument('--bs',                default=112 ,     type=int,   help='Mini-Batchsize to use.')
parser.add_argument('--samples_per_class', default=4,        type=int,   help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
parser.add_argument('--seed',              default=1,        type=int,   help='Random seed for reproducibility.')
parser.add_argument('--scheduler',         default='step',   type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
parser.add_argument('--gamma',             default=0.3,      type=float, help='Learning rate reduction after tau epochs.')
parser.add_argument('--decay',             default=0.0004,   type=float, help='Weight decay for optimizer.')
parser.add_argument('--tau',               default=[30,55],nargs='+',type=int,help='Stepsize(s) before reducing learning rate.')

##### Loss-specific Settings
parser.add_argument('--loss',         default='marginloss', type=str,   help='loss options: marginloss, triplet, npair, proxynca')
parser.add_argument('--sampling',     default='distance',   type=str,   help='For triplet-based losses: Modes of Sampling: random, semihard, distance.')
### MarginLoss
parser.add_argument('--margin',       default=0.2,          type=float, help='TRIPLET/MARGIN: Margin for Triplet-based Losses')
parser.add_argument('--beta_lr',      default=0.0005,       type=float, help='MARGIN: Learning Rate for class margin parameters in MarginLoss')
parser.add_argument('--beta',         default=1.2,          type=float, help='MARGIN: Initial Class Margin Parameter in Margin Loss')
parser.add_argument('--nu',           default=0,            type=float, help='MARGIN: Regularisation value on betas in Margin Loss.')
parser.add_argument('--beta_constant',                      action='store_true', help='MARGIN: Use constant, un-trained beta.')
### ProxyNCA
parser.add_argument('--proxy_lr',     default=0.00001,     type=float, help='PROXYNCA: Learning Rate for Proxies in ProxyNCALoss.')
### NPair L2 Penalty
parser.add_argument('--l2npair',      default=0.02,        type=float, help='NPAIR: Penalty-value for non-normalized N-PAIR embeddings.')

##### Evaluation Settings
parser.add_argument('--k_vals',       nargs='+', default=[1,2,4,8], type=int, help='Recall @ Values.')

##### Network parameters
parser.add_argument('--embed_dim',    default=128,         type=int,   help='Embedding dimensionality of the network. Note: in literature, dim=128 is used for ResNet50 and dim=512 for GoogLeNet.')
parser.add_argument('--arch',         default='resnet50',  type=str,   help='Network backend choice: resnet50, googlenet.')
parser.add_argument('--not_pretrained',                    action='store_true', help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')
parser.add_argument('--grad_measure',                      action='store_true', help='If added, gradients passed from embedding layer to the last conv-layer are stored in each iteration.')
parser.add_argument('--dist_measure',                      action='store_true', help='If added, the ratio between intra- and interclass distances is stored after each epoch.')

##### Setup Parameters
parser.add_argument('--gpu',          default=0,           type=int,   help='GPU-id for GPU to use.')
parser.add_argument('--savename',     default='',          type=str,   help='Save folder name if any special information is to be included.')

### Paths to datasets and storage folder
parser.add_argument('--source_path',  default=os.getcwd()+'/Datasets',         type=str, help='Path to training data.')
parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save everything.')

##### Read in parameters
opt = parser.parse_args()


"""============================================================================"""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset

if opt.dataset=='online_products':
    opt.k_vals = [1,10,100,1000]
if opt.dataset=='in-shop':
    opt.k_vals = [1,10,20,30,50]
if opt.dataset=='vehicle_id':
    opt.k_vals = [1,5]

if opt.loss == 'proxynca':
    opt.samples_per_class = 1
else:
    assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

if opt.loss == 'npair' or opt.loss == 'proxynca': opt.sampling = 'None'

opt.pretrained = not opt.not_pretrained


"""============================================================================"""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)


"""============================================================================"""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True
np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)


"""============================================================================"""
##################### NETWORK SETUP ##################
opt.device = torch.device('cuda')
#Depending on the choice opt.arch, networkselect() returns the respective network model
model      = netlib.networkselect(opt)

print('{} Setup for {} with {} sampling on {} complete with #weights: {}'.format(opt.loss.upper(), opt.arch.upper(), opt.sampling.upper(), opt.dataset.upper(), aux.gimme_params(model)))

#Push to Device
_          = model.to(opt.device)
#Place trainable parameter in list of parameters to train:
if 'fc_lr_mul' in vars(opt).keys() and opt.fc_lr_mul!=0:
    all_but_fc_params = list(filter(lambda x: 'last_linear' not in x[0],model.named_parameters()))
    fc_params         = model.model.last_linear.parameters()
    to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
                         {'params':fc_params,'lr':opt.lr*opt.fc_lr_mul,'weight_decay':opt.decay}]
else:
    to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]




"""============================================================================"""
#################### DATALOADER SETUPS ##################
#Returns a dictionary containing 'training', 'testing', and 'evaluation' dataloaders.
#The 'testing'-dataloader corresponds to the validation set, and the 'evaluation'-dataloader
#Is simply using the training set, however running under the same rules as 'testing' dataloader,
#i.e. no shuffling and no random cropping.
dataloaders      = data.give_dataloaders(opt.dataset, opt)
#Because the number of supervised classes is dataset dependent, we store them after
#initializing the dataloader
opt.num_classes  = len(dataloaders['training'].dataset.avail_classes)




"""============================================================================"""
#################### CREATE LOGGING FILES ###############
#Each dataset usually has a set of standard metrics to log. aux.metrics_to_examine()
#returns a dict which lists metrics to log for training ('train') and validation/testing ('val')

metrics_to_log = aux.metrics_to_examine(opt.dataset, opt.k_vals)
# example output: {'train': ['Epochs', 'Time', 'Train Loss', 'Time'],
#                  'val': ['Epochs','Time','NMI','F1', 'Recall @ 1','Recall @ 2','Recall @ 4','Recall @ 8']}

#Using the provided metrics of interest, we generate a LOGGER instance.
#Note that 'start_new' denotes that a new folder should be made in which everything will be stored.
#This includes network weights as well.
LOG = aux.LOGGER(opt, metrics_to_log, name='Base', start_new=True)
#If graphviz is installed on the system, a computational graph of the underlying
#network will be made as well.
try:
    aux.save_graph(opt, model)
except:
    print('Cannot generate graph!')



"""============================================================================"""
##################### OPTIONAL EVALUATIONS #####################
#Store the averaged gradients returned from the embedding to the last conv. layer.
if opt.grad_measure:
    grad_measure = eval.GradientMeasure(opt, name='baseline')
#Store the relative distances between average intra- and inter-class distance.
if opt.dist_measure:
    #Add a distance measure for training distance ratios
    distance_measure = eval.DistanceMeasure(dataloaders['evaluation'], opt, name='Train', update_epochs=1)
    # #If uncommented: Do the same for the test set
    # distance_measure_test = eval.DistanceMeasure(dataloaders['testing'], opt, name='Train', update_epochs=1)


"""============================================================================"""
#################### LOSS SETUP ####################
#Depending on opt.loss and opt.sampling, the respective criterion is returned,
#and if the loss has trainable parameters, to_optim is appended.
criterion, to_optim = losses.loss_select(opt.loss, opt, to_optim)
_ = criterion.to(opt.device)

"""============================================================================"""
#################### OPTIM SETUP ####################
#As optimizer, Adam with standard parameters is used.
optimizer    = torch.optim.Adam(to_optim)

if opt.scheduler=='exp':
    scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
elif opt.scheduler=='step':
    scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
elif opt.scheduler=='none':
    print('No scheduling used!')
else:
    raise Exception('No scheduling option for input: {}'.format(opt.scheduler))



"""============================================================================"""
#################### TRAINER FUNCTION ############################
def train_one_epoch(train_dataloader, model, optimizer, criterion, opt, epoch):
    """
    This function is called every epoch to perform training of the network over one full
    (randomized) iteration of the dataset.

    Args:
        train_dataloader: torch.utils.data.DataLoader, returns (augmented) training data.
        model:            Network to train.
        optimizer:        Optimizer to use for training.
        criterion:        criterion to use during training.
        opt:              argparse.Namespace, Contains all relevant parameters.
        epoch:            int, Current epoch.

    Returns:
        Nothing!
    """
    loss_collect = []

    start = time.time()

    data_iterator = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))
    for i,(class_labels, input) in enumerate(data_iterator):
        #Compute embeddings for input batch.
        features  = model(input.to(opt.device))
        #Compute loss.
        loss      = criterion(features, class_labels)

        #Ensure gradients are set to zero at beginning
        optimizer.zero_grad()
        #Compute gradients.
        loss.backward()

        if opt.grad_measure:
            #If desired, save computed gradients.
            grad_measure.include(model.model.last_linear)

        #Update weights using comp. gradients.
        optimizer.step()

        #Store loss per iteration.
        loss_collect.append(loss.item())
        if i==len(train_dataloader)-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))

    #Save metrics
    LOG.log('train', LOG.metrics_to_log['train'], [epoch, np.round(time.time()-start,4), np.mean(loss_collect)])

    if opt.grad_measure:
        #Dump stored gradients to Pickle-File.
        grad_measure.dump(epoch)




"""============================================================================"""
"""========================== MAIN TRAINING PART =============================="""
"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')
for epoch in range(opt.n_epochs):
    ### Print current learning rates for all parameters
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    ### Train one epoch
    _ = model.train()
    train_one_epoch(dataloaders['training'], model, optimizer, criterion, opt, epoch)

    ### Evaluate
    _ = model.eval()
    #Each dataset requires slightly different dataloaders.
    if opt.dataset in ['cars196', 'cub200', 'online_products']:
        eval_params = {'dataloader':dataloaders['testing'], 'model':model, 'opt':opt, 'epoch':epoch}
    elif opt.dataset=='in-shop':
        eval_params = {'query_dataloader':dataloaders['testing_query'], 'gallery_dataloader':dataloaders['testing_gallery'], 'model':model, 'opt':opt, 'epoch':epoch}
    elif opt.dataset=='vehicle_id':
        eval_params = {'dataloaders':[dataloaders['testing_set1'], dataloaders['testing_set2'], dataloaders['testing_set3']], 'model':model, 'opt':opt, 'epoch':epoch}

    #Compute Evaluation metrics, print them and store in LOG.
    eval.evaluate(opt.dataset, LOG, save=True, **eval_params)

    #Update the Metric Plot and save it.
    LOG.update_info_plot()

    #(optional) compute ratio of intra- to interdistances.
    if opt.dist_measure:
        distance_measure.measure(model, epoch)
        # distance_measure_test.measure(model, epoch)

    ### Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()

    print('\n-----\n')
