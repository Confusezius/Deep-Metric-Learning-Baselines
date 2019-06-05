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

import torch, random, itertools as it, numpy as np, faiss, random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from PIL import Image

import numpy as np



"""================================================================================================="""
def loss_select(loss, opt, to_optim):
    if loss=='triplet':
        loss_params  = {'margin':opt.margin, 'sampling_method':opt.sampling}
        criterion    = TripletLoss(**loss_params)
    elif loss=='npair':
        loss_params  = {'l2':opt.l2npair}
        criterion    = NPairLoss(**loss_params)
    elif loss=='marginloss':
        loss_params  = {'margin':opt.margin, 'nu': opt.nu, 'beta':opt.beta, 'n_classes':opt.num_classes, 'sampling_method':opt.sampling}
        criterion    = MarginLoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.beta_lr, 'weight_decay':0}]
    elif loss=='proxynca':
        loss_params  = {'num_proxies':opt.num_classes, 'embedding_dim':opt.classembed if 'num_cluster' in vars(opt).keys() else opt.embed_dim}
        criterion    = ProxyNCALoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.proxy_lr}]
    elif loss=='crossentropy':
        loss_params  = {'n_classes':opt.num_classes, 'inp_dim':opt.embed_dim}
        criterion    = CEClassLoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.lr, 'weight_decay':0}]
    else:
        raise Exception('Loss {} not available!'.format(loss))

    return criterion, to_optim





"""================================================================================================="""
### Sampler() holds all possible triplet sampling options: random, SemiHardNegative & Distance-Weighted.
class Sampler():
    def __init__(self, method='random'):
        self.method = method
        if method=='semihard':
            self.give = self.semihardsampling
        elif method=='distance':
            self.give = self.distanceweightedsampling
        elif method=='npair':
            self.give = self.npairsampling
        elif method=='random':
            self.give = self.randomsampling

    def randomsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and randomly
        selects batch.batchsize triplets.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        unique_classes = np.unique(labels)
        indices        = np.arange(len(batch))
        class_dict     = {i:indices[labels==i] for i in unique_classes}

        sampled_triplets = [list(it.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        #NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
        return sampled_triplets


    def semihardsampling(self, batch, labels):
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            anchors.append(i)
            #1 for batchelements with label l
            neg = labels!=l; pos = labels==l
            #0 for current anchor
            pos[i] = False

            #Find negatives that violate triplet constraint semi-negatives
            neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
            #Find positives that violate triplet constraint semi-hardly
            pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())

            if pos_mask.sum()>0:
                positives.append(np.random.choice(np.where(pos_mask)[0]))
            else:
                positives.append(np.random.choice(np.where(pos)[0]))

            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets


    def distanceweightedsampling(self, batch, labels, lower_cutoff=0.5, upper_cutoff=1.4):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)



        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))
            #Sample negatives by distance
            negatives.append(np.random.choice(bs,p=q_d_inv))

        sampled_triplets = [[a,p,n] for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets


    def npairsampling(self, batch, labels, gt_labels=None):
        if isinstance(labels, torch.Tensor):    labels = labels.detach().cpu().numpy()

        label_set, count = np.unique(labels, return_counts=True)
        label_set  = label_set[count>=2]
        pos_pairs  = np.array([np.random.choice(np.where(labels==x)[0], 2, replace=False) for x in label_set])
        neg_tuples = []

        for idx in range(len(pos_pairs)):
            neg_tuples.append(pos_pairs[np.delete(np.arange(len(pos_pairs)),idx),1])

        neg_tuples = np.array(neg_tuples)

        sampled_npairs = [[a,p,*list(neg)] for (a,p),neg in zip(pos_pairs, neg_tuples)]
        return sampled_npairs


    def pdist(self, A, eps = 1e-4):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = eps).sqrt()


    def inverse_sphere_distances(self, batch, dist, labels, anchor_label):
        bs,dim       = len(dist),batch.shape[-1]

        #negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
        log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

        q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
        q_d_inv[np.where(labels==anchor_label)[0]] = 0

        ### NOTE: Cutting of values with high distances made the results slightly worse.
        # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

        q_d_inv = q_d_inv/q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()




"""================================================================================================="""
### Standard Triplet Loss, finds triplets in Mini-batches.
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1, sampling_method='random', size_average=False):
        """
        Args:
            margin:             Triplet Margin.
            triplets_per_batch: A batch allows for multitudes of triplets to use. This gives the number
                                if triplets to sample from.
        """
        super(TripletLoss, self).__init__()
        self.margin             = margin
        self.size_average       = size_average
        self.sampler            = Sampler(method=sampling_method)

    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels, gt_labels=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
            sampled_triplets: Optional: Provided pre-sampled triplets
        """
        if gt_labels is not None:
            sampled_triplets = self.sampler.give(batch, labels, gt_labels)
        else:
            sampled_triplets = self.sampler.give(batch, labels)
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        if self.size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)



"""================================================================================================="""
### Standard N-Pair Loss.
class NPairLoss(torch.nn.Module):
    def __init__(self, l2=0.02):
        """
        Args:
            margin:             Triplet Margin.
            triplets_per_batch: A batch allows for multitudes of triplets to use. This gives the number
                                if triplets to sample from.
        """
        super(NPairLoss, self).__init__()
        self.sampler = Sampler(method='npair')
        self.l2      = l2

    def npair_distance(self, anchor, positive, negatives):
        return torch.log(1+torch.sum(torch.exp(anchor.mm((negatives-positive).transpose(0,1)))))

    def l2dist(self, anchor, positive):
        ### Only need to penalize anchor and positive since the negative contain only those
        return torch.sum(anchor**2+positive**2)

    def forward(self, batch, labels, gt_labels=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        if gt_labels is not None:
            sampled_npairs = self.sampler.give(batch, labels, gt_labels)
        else:
            sampled_npairs = self.sampler.give(batch, labels)

        # from IPython import embed
        # embed()
        loss           = torch.stack([self.npair_distance(batch[npair[0]:npair[0]+1,:],batch[npair[1]:npair[1]+1,:],batch[npair[2:],:]) for npair in sampled_npairs])
        loss           = loss + self.l2*torch.mean(torch.stack([self.l2dist(batch[npair[0],:], batch[npair[1],:]) for npair in sampled_npairs]))
        return torch.mean(loss)




"""================================================================================================="""
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class MarginLoss(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance'):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(MarginLoss, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant

        self.beta_val = beta
        self.beta = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampling_method    = sampling_method
        self.sampler            = Sampler(method=sampling_method)


    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        sampled_triplets = self.sampler.give(batch, labels)

        d_ap, d_an = [],[]
        for triplet in sampled_triplets:
            train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

            pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
            neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

        #
        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        #
        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        #
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        #
        loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count

        #
        if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss





"""================================================================================================="""
### ProxyNCALoss containing trainable class proxies. Works independent of batch size.
class ProxyNCALoss(torch.nn.Module):
    def __init__(self, num_proxies, embedding_dim):
        """
        Implementation for PROXY-NCA loss. Note that here, Proxies are part of the loss, not the network.
        Args:
            num_proxies:    num_proxies is the number of Class Proxies to use. Normally equals the number of classes.
            embedding_dim:  Feature dimensionality. Proxies share the same dim. as the embeddings.
        """
        super(ProxyNCALoss, self).__init__()
        self.num_proxies   = num_proxies
        self.embedding_dim = embedding_dim
        self.PROXIES = torch.nn.Parameter(torch.randn(num_proxies, self.embedding_dim) / 8)
        self.all_classes = torch.arange(num_proxies)


    def forward(self, anchor_batch, classes):
        """
        Args:
            anchor_batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            classes: nparray/list: For each element of the anchor_batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        anchor_batch = 3*torch.nn.functional.normalize(anchor_batch, dim=1)
        PROXIES      = 3*torch.nn.functional.normalize(self.PROXIES, dim=1)
        pos_proxies = torch.stack([PROXIES[pos_label:pos_label+1,:] for pos_label in classes])
        neg_proxies = torch.stack([torch.cat([self.all_classes[:class_label],self.all_classes[class_label+1:]]) for class_label in classes])
        neg_proxies = torch.stack([PROXIES[neg_labels,:] for neg_labels in neg_proxies])

        dist_to_neg_proxies = torch.sum((anchor_batch[:,None,:]-neg_proxies).pow(2),dim=-1)
        dist_to_pos_proxies = torch.sum((anchor_batch[:,None,:]-pos_proxies).pow(2),dim=-1)

        negative_log_proxy_nca_loss = torch.mean(dist_to_pos_proxies[:,0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))
        return negative_log_proxy_nca_loss





"""================================================================================================="""
class CEClassLoss(torch.nn.Module):
    def __init__(self, inp_dim, n_classes):
        super(CEClassLoss, self).__init__()
        self.mapper  = torch.nn.Sequential(torch.nn.Linear(inp_dim, n_classes))
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, aux_features, labels):
        return self.ce_loss(self.mapper(aux_features), labels.type(torch.cuda.LongTensor))
