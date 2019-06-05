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

import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json, csv
import matplotlib
import matplotlib.pyplot as plt

from scipy.spatial import distance
from sklearn.preprocessing import normalize

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn
import auxiliaries as aux
import datasets as data

import netlib
import losses as losses

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


"""=================================================================================================================="""
"""=================================================================================================================="""
"""========================================================="""
def evaluate(dataset, LOG, **kwargs):
    if dataset in ['cars196', 'cub200', 'online_products']:
        ret = evaluate_one_dataset(LOG, **kwargs)
    elif dataset in ['in-shop']:
        ret = evaluate_query_and_gallery_dataset(LOG, **kwargs)
    elif dataset in ['vehicle_id']:
        ret = evaluate_multiple_datasets(LOG, **kwargs)
    else:
        raise Exception('No implementation for dataset {} available!')

    return ret


"""========================================================="""
class DistanceMeasure():
    def __init__(self, checkdata, opt, name='Train', update_epochs=1):
        """
        Args:
            checkdata:
            opt:
            name:
            update_epochs:
        Returns:
            Nothing!
        """
        self.update_epochs = update_epochs
        self.pars          = opt
        self.save_path = opt.save_path

        self.name          = name
        self.csv_file      = opt.save_path+'/distance_measures_{}.csv'.format(self.name)
        with open(self.csv_file,'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['Rel. Intra/Inter Distance'])

        self.checkdata     = checkdata

        self.mean_class_dists = []
        self.epochs           = []


    def measure(self, model, epoch):
        """
        Args:
            model:
            epoch:
        Returns:
            Nothing!
        """
        if epoch%self.update_epochs: return

        self.epochs.append(epoch)

        torch.cuda.empty_cache()

        _ = model.eval()

        with torch.no_grad():
            feature_coll, target_coll = [],[]
            data_iter = tqdm(self.checkdata, desc='Estimating Data Distances...')
            for idx, data in enumerate(data_iter):
                input_img, target = data[1], data[0]
                features = model(input_img.to(self.pars.device))
                feature_coll.extend(features.cpu().detach().numpy().tolist())
                target_coll.extend(target.numpy().tolist())

        feature_coll = np.vstack(feature_coll).astype('float32')
        target_coll   = np.hstack(target_coll).reshape(-1)
        avail_labels  = np.unique(target_coll)

        class_positions = []
        for lab in avail_labels:
            class_positions.append(np.where(target_coll==lab)[0])

        com_class, dists_class = [],[]
        for class_pos in class_positions:
            dists = distance.cdist(feature_coll[class_pos],feature_coll[class_pos],'cosine')
            dists = np.sum(dists)/(len(dists)**2-len(dists))
            # dists = np.linalg.norm(np.std(feature_coll_aux[class_pos],axis=0).reshape(1,-1)).reshape(-1)
            com   = normalize(np.mean(feature_coll[class_pos],axis=0).reshape(1,-1)).reshape(-1)
            dists_class.append(dists)
            com_class.append(com)

        mean_inter_dist = distance.cdist(np.array(com_class), np.array(com_class), 'cosine')
        mean_inter_dist = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))

        mean_class_dist = np.mean(np.array(dists_class)/mean_inter_dist)
        self.mean_class_dists.append(mean_class_dist)

        self.update(mean_class_dist)


    def update(self, mean_class_dist):
        """
        Args:
            mean_class_dist:
        Returns:
            Nothing!
        """
        self.update_csv(mean_class_dist)
        self.update_plot(mean_class_dist)


    def update_csv(self, mean_class_dist):
        """
        Args:
            mean_class_dist:
        Returns:
            Nothing!
        """
        with open(self.csv_file, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([mean_class_dist])


    def update_plot(self, mean_class_dist):
        """
        Args:
            mean_class_dist:
        Returns:
            Nothing!
        """
        plt.style.use('ggplot')
        f,ax = plt.subplots(1)
        ax.set_title('Mean Intra- over Interclassdistances')
        ax.plot(self.epochs, self.mean_class_dists, label='Class')
        f.legend()
        f.set_size_inches(15,8)
        f.savefig(self.save_path+'/distance_measures_{}.svg'.format(self.name))




class GradientMeasure():
    def __init__(self, opt, name='class-it'):
        """
        Args:
            opt:
            name:
        Returns:
            Nothing!
        """
        self.pars  = opt
        self.name  = name
        self.saver = {'grad_normal_mean':[], 'grad_normal_std':[], 'grad_abs_mean':[], 'grad_abs_std':[]}


    def include(self, params):
        """
        Args:
            params:
        Returns:
            Nothing!
        """
        gradients = [params.weight.grad.detach().cpu().numpy()]

        for grad in gradients:
            ### Shape: 128 x 2048
            self.saver['grad_normal_mean'].append(np.mean(grad,axis=0))
            self.saver['grad_normal_std'].append(np.std(grad,axis=0))
            self.saver['grad_abs_mean'].append(np.mean(np.abs(grad),axis=0))
            self.saver['grad_abs_std'].append(np.std(np.abs(grad),axis=0))

    def dump(self, epoch):
        """
        Args:
            epoch:
        Returns:
            Nothing!
        """
        with open(self.pars.save_path+'/grad_dict_{}.pkl'.format(self.name),'ab') as f:
            pkl.dump([self.saver], f)
        self.saver = {'grad_normal_mean':[], 'grad_normal_std':[], 'grad_abs_mean':[], 'grad_abs_std':[]}




"""========================================================="""
def evaluate_one_dataset(LOG, dataloader, model, opt, save=True, give_return=False, weights=[], epoch=0):
    start = time.time()
    image_paths = np.array(dataloader.dataset.image_list)
    with torch.no_grad():
        # model_generated_cluster_labels, target_labels, feature_coll, computed_centroids = aux.eval_metrics_one_dataset(model, dataloader, device=opt.device, spliteval=spliteval, k_vals=opt.k_vals, opt=opt, evaltype=evaltype, weights=weights)
        # return model_generated_cluster_labels, target_labels, feature_coll, computed_centroids
        F1, NMI, recall_at_ks, feature_matrix_all = aux.eval_metrics_one_dataset(model, dataloader, device=opt.device, k_vals=opt.k_vals, opt=opt)
        result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(opt.k_vals, recall_at_ks))
        result_str = 'Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(epoch, NMI, F1, result_str)

        if LOG is not None:
            if save:
                if not len(LOG.progress_saver['val']['Recall @ 1']) or recall_at_ks[0]>np.max(LOG.progress_saver['val']['Recall @ 1']):
                    aux.set_checkpoint(model, opt, LOG.progress_saver, LOG.prop.save_path+'/checkpoint.pth.tar')
                    aux.recover_closest_one_dataset(feature_matrix_all, image_paths, LOG.prop.save_path+'/sample_recoveries.png')

            LOG.log('val', LOG.metrics_to_log['val'], [epoch, np.round(time.time()-start), NMI, F1]+recall_at_ks)

    print(result_str)
    if give_return:
        return recall_at_ks, NMI, F1
    else:
        None




"""========================================================="""
def evaluate_query_and_gallery_dataset(LOG, query_dataloader, gallery_dataloader, model, opt, save=True, give_return=False, epoch=0):
    start = time.time()
    query_image_paths   = np.array([x[0] for x in query_dataloader.dataset.image_list])
    gallery_image_paths = np.array([x[0] for x in gallery_dataloader.dataset.image_list])

    with torch.no_grad():
        F1, NMI, recall_at_ks, query_feature_matrix_all, gallery_feature_matrix_all = aux.eval_metrics_query_and_gallery_dataset(model, query_dataloader, gallery_dataloader, device=opt.device, k_vals = opt.k_vals, opt=opt)

        result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(opt.k_vals, recall_at_ks))
        result_str = 'Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(epoch, NMI, F1, result_str)

        if LOG is not None:
            if save:
                if not len(LOG.progress_saver['val']['Recall @ 1']) or recall_at_ks[0]>np.max(LOG.progress_saver['val']['Recall @ 1']):
                    aux.set_checkpoint(model, opt, LOG.progress_saver, LOG.prop.save_path+'/checkpoint.pth.tar')
                    aux.recover_closest_inshop(query_feature_matrix_all, gallery_feature_matrix_all, query_image_paths, gallery_image_paths, LOG.prop.save_path+'/sample_recoveries.png')

            LOG.log('val', LOG.metrics_to_log['val'], [epoch, np.round(time.time()-start), NMI, F1]+recall_at_ks)

    print(result_str)
    if give_return:
        return recall_at_ks, NMI, F1
    else:
        None






"""========================================================="""
def evaluate_multiple_datasets(LOG, dataloaders, model, opt, save=True, give_return=False, weights=[], epoch=0):
    start = time.time()

    csv_data = [epoch]

    with torch.no_grad():
        for i,dataloader in enumerate(dataloaders):
            image_paths = np.array(dataloader.dataset.image_list)
            F1, NMI, recall_at_ks, feature_matrix_all = aux.eval_metrics_one_dataset(model, dataloader, device=opt.device, k_vals=opt.k_vals, opt=opt)

            result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(opt.k_vals, recall_at_ks))
            result_str = 'SET {}: Epoch (Test) {0}: NMI [{1:.4f}] | Recall [{2}]'.format(i+1, epoch, NMI, result_str)

            if LOG is not None:
                if save:
                    if not len(LOG.progress_saver['Recall @ 1']) or recall_at_ks[0]>np.max(LOG.progress_saver['Recall @ 1']):
                        aux.set_checkpoint(model, opt, LOG.progress_saver, LOG.prop.save_path+'/checkpoint_set{}.pth.tar'.format(i+1))
                        aux.recover_closest_one_dataset(feature_matrix_all, image_paths, LOG.prop.save_path+'/sample_recoveries_set{}.png'.format(i+1))

                csv_data += [NMI, F1]+recall_at_ks

    print(result_str)
    csv_data.insert(0, np.round(time.time()-start))
    LOG.csv_loggers['val'].log('val', LOG.metrics_to_log['val'], csv_data)
