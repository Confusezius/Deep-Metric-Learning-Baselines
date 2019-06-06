# Easily Extendable Basic Deep Metric Learning Pipeline
### ___Authors___: Karsten Roth (karsten.rh1@gmail.com), Biagio Brattoli (biagio.brattoli@gmail.com)

---
### FOR USAGE, GO TO SECTION 3

## 1. Overview
This repository contains a full, easily extendable pipeline to test and implement current and new deep metric learning methods. For referencing and testing, this repo contains implementations/dataloaders for:

__Loss Functions__
* Triplet Loss (https://arxiv.org/abs/1412.6622)
* Margin Loss (https://arxiv.org/abs/1706.07567)
* ProxyNCA (https://arxiv.org/abs/1703.07464)
* N-Pair Loss (https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)

__Sampling Methods__
* Random Sampling
* Semihard Sampling (https://arxiv.org/abs/1511.06452)
* Distance Sampling (https://arxiv.org/abs/1706.07567)
* N-Pair Sampling (https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)

__Datasets__
* CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)
* CARS196 (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)
* (Optional) In-Shop Clothes (http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)
* (optional) PKU Vehicle-ID (https://www.pkuml.org/resources/pku-vds.html)

__Architectures__
* GoogLeNet (https://arxiv.org/abs/1409.4842)
* ResNet50 (https://arxiv.org/pdf/1512.03385.pdf)

__NOTE__: In-Shop Clothes and PKU Vehicle-ID are _(optional)_ because there is no direct way to download the dataset. The former webpage has a broken download link, and the latter requires special licensing. However, if these datasets are available (in the structure shown in part 2.2), they can be used directly.


---

## 2. Repo & Dataset Structure
### 2.1 Repo Structure
```
Repository
│   ### General Files
│   README.md
│   requirements.txt    
│   installer.sh
|
|   ### Main Scripts
|   main.py     (main training script)
|   losses.py   (collection of loss and sampling impl.)
│   datasets.py (dataloaders for all datasets)
│   
│   ### Utility scripts
|   auxiliaries.py  (set of useful utilities)
|   evaluate.py     (set of evaluation functions)
│   
│   ### Network Scripts
|   netlib.py       (contains impl. for ResNet50)
|   googlenet.py    (contains impl. for GoogLeNet)
│   
│   
└───Training Results (generated during Training)
|    │   e.g. cub200/Training_Run_Name
|    │   e.g. cars196/Training_Run_Name
|
│   
└───Datasets (should be added, if one does not want to set paths)
|    │   cub200
|    │   cars196
|    │   online_products
|    │   in-shop
|    │   vehicle_id
```

### 2.2 Dataset Structures
__CUB200-2011__
```
cub200
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
```

__CARS196__
```
cars196
└───images
|    └───Acura Integra Type R 2001
|           │   00128.jpg
|           │   ...
|    ...
```

__Online Products__
```
online_products
└───images
|    └───bicycle_final
|           │   111085122871_0.jpg
|    ...
|
└───Info_Files
|    │   bicycle.txt
|    │   ...
```

__In-Shop Clothes__
```
in-shop
└───img
|    └───MEN
|         └───Denim
|               └───id_00000080
|                       │   01_1_front.jpg
|                       │   ...
|               ...
|         ...
|    ...
|
└───Eval
|    │   list_eval_partition.txt
```


__PKU Vehicle ID__
```
vehicle_id
└───image
|     │   <img>.jpg
|     |   ...
|     
└───train_test_split
|     |   test_list_800.txt
|     |   ...
```

---

## 3. Using the Pipeline

### [1.] Requirements
The pipeline is build around `Python3` (i.e. by installing Miniconda https://conda.io/miniconda.html') and `Pytorch 1.0.0/1`. It has been tested around `cuda 8` and `cuda 9`.

To install the required libraries, either directly check `requirements.txt` or create a conda environment:
```
conda create -n <Env_Name> python=3.6
```

Activate it
```
conda activate <Env_Name>
```
and run
```
bash installer.sh
```

Note that for kMeans- and Nearest Neighbour Computation, the library `faiss` is used, which can allow to move these computations to GPU if speed is desired. However, in most cases, `faiss` is fast enough s.t. the computation of evaluation metrics is no bottleneck.



### [2.] Exemplary Runs
The main script is `main.py`. If running without input arguments, training of ResNet50 on CUB200-2011 with Marginloss and Distance-sampling is performed.  
Otherwise, the following flags suffice to train with different losses, sampling methods, architectures and datasets:
```
python main.py --dataset <dataset> --loss <loss> --sampling <sampling> --arch <arch> --k_vals <k_vals> --embed_dim <embed_dim>
```
The following flags are available:
* `<dataset> <- cub200, cars196, online_products, in-shop, vehicle_id`
* `<loss> <- marginloss, triplet, npair, proxynca`
* `<sampling> <- distance, semihard, random, npair`
* `<arch> <- resnet50, googlenet`
* `<k_vals> <- List of Recall @ k values to evaluate on, e.g. 1 2 4 8`
* `<embed_dim> <- Network embedding dimension. Default: 128 for ResNet50, 512 for GoogLeNet.`

For all other training-specific arguments (e.g. batch-size, num. training epochs., ...), simply refer to the input arguments in `main.py`.

Finally, to decide the GPU to use and the name of the training folder in which network weights, sample recoveries and metrics are stored, set:
```
python main.py --gpu <gpu_id> --savename <name_of_training_run>
```
If `--savename` is not set, a default name based on the starting date will be chosen.

If one wishes to simply use standard parameters and wants to get close to literature results (more or less, depends on seeds and overall training scheduling), refer to `sample_training_runs.sh`, which contains a list of executable one-liners.


### [3.] Implementation Notes regarding Extendability:

To extend or test other sampling or loss methods, simply do:

__For Batch-based Sampling:__  
In `losses.py`, add the sampling method, which should act on a batch (and the resp. set of labels), e.g.:
```
def new_sampling(self, batch, label, **additional_parameters): ...
```
This function should, if it needs to run with existing losses, a list of tuples containing indexes with respect to the batch, e.g. for sampling methods returning triplets:
```
return [(anchor_idx, positive_idx, negative_idx) for anchor_idx, positive_idx, negative_idx in zip(anchor_idxs, positive_idxs, negative_idxs)]
```
Also, don't forget to add a handle in `Sampler.__init__()`.

__For Data-specific Sampling:__  
To influence the data samples used to generate the batches, in `datasets.py` edit `BaseTripletDataset`.


__For New Loss Functions:__  
Simply add a new class inheriting from `torch.nn.Module`. Refer to other loss variants to see how to do so. In general, include an instance of the `Sampler`-class, which will provide sampled data tuples during a `forward()`-pass, by calling `self.sampler_instance.give(batch, labels, **additional_parameters)`.  
Finally, include the loss function in the `loss_select()`-function. Parameters can be passed through the dictionary-notation (see other examples) and if learnable parameters are added, include them in the `to_optim`-list.

### [4.] Additional Notes:
To finalize, several flags might be of interest when examining the respective runs:
```
--dist_measure: If set, the ratio of mean intraclass-distances over mean interclass distances
                (by measure of center-of-mass distances) is computed after each epoch and stored/plotted.
--grad_measure: If set, the average (absolute) gradients from the embedding layer to the last
                conv. layer are stored in a Pickle-File. This can be used to examine the change of features during each iteration.
```
For more details, refer to the respective classes in `auxiliaries.py`.

---

## 4. Results

__CUB200__

Architecture | Loss/Sampling |   NMI  |  F1  | Recall @ 1,2,4,8
-------------|---------------|--------|------|-----------------
ResNet50     |               |        |      |     
GoogLeNet    |               |        |      |      

__Cars196__

Architecture | Loss/Sampling |   NMI  |  F1  | Recall @ 1,2,4,8
-------------|---------------|--------|------|-----------------
ResNet50     |               |        |      |     
GoogLeNet    |               |        |      |      

__Online Products__

Architecture | Loss/Sampling |   NMI  |  F1  | Recall @ 1,10,100,1000
-------------|---------------|--------|------|-----------------
ResNet50     |               |        |      |     
GoogLeNet    |               |        |      |      

__In-Shop Clothes__

Architecture | Loss/Sampling |   NMI  |  F1  | Recall @ 1,10,20,30,50
-------------|---------------|--------|------|-----------------
ResNet50     |               |        |      |     
GoogLeNet    |               |        |      |      


__Vehicle ID__ *(medium test set)*

Architecture | Loss/Sampling |   NMI  |  F1  | Recall @ 1,5
-------------|---------------|--------|------|-----------------
ResNet50     |               |        |      |     
GoogLeNet    |               |        |      |      


---


## ToDO:
- [x] Fix Version in `requirements.txt`  
- [ ] Add Results for Implementations
- [x] Finalize Comments  
- [ ] Add Inception-BN  
- [ ] Add Lifted Structure Loss
