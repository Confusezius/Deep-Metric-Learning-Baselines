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

Architecture | Loss/Sampling       |   NMI  |  F1  | Recall @ 1,2,4,8
-------------|---------------      |--------|------|-----------------
ResNet50     |  Margin/Distance    | __68.2__   | __38.7__ | 63.4, 74.9, __86.0__, 90.4    
ResNet50     |  Triplet/Semihard   | 66.4   | 35.3 | 61.4, 73.3, 82.7, 89.6    
ResNet50     |  NPair/None         | 65.4   | 33.8 | 59.0, 71.3, 81.1, 88.8    
ResNet50     |  ProxyNCA/None      | 68.1   | 38.1 | __64.0__, __75.4__, 84.2, __90.5__    
 -           |         -           |    -   |   -  |  -   
GoogLeNet    |  Margin/Distance    | __62.5__   | __31.9__ | __57.9, 69.7, 79.9, 87.7__    
GoogLeNet    |  Triplet/Semihard   | 61.6   | 29.7 | 56.8, 68.9, 78.7, 86.7    
GoogLeNet    |  NPair/None         | 59.2   | 26.2 | 50.6, 63.3, 74.5, 83.7    
GoogLeNet    |  ProxyNCA/None      | 61.2   | 29.0 | 55.4, 67.3, 77.8, 85.1    


__Cars196__

Architecture | Loss/Sampling       |   NMI  |  F1  | Recall @ 1,2,4,8
-------------|---------------      |--------|------|-----------------
ResNet50     |  Margin/Distance    | __67.2__   | __37.6__ | 79.3, 87.1, __92.1, 95.4__    
ResNet50     |  Triplet/Semihard   | 64.2   | 32.7 | 75.2, 84.1, 90.0, 94.0
ResNet50     |  NPair/None         | 62.3   | 30.1 | 69.5, 80.2, 87.3, 92.1
ResNet50     |  ProxyNCA/None      | 66.3   | 35.8 | __80.0, 87.2__, 91.8, 95.1
 -           |         -           |    -   |   -  |  -   
GoogLeNet    |  Margin/Distance    
GoogLeNet    |  Triplet/Semihard   
GoogLeNet    |  NPair/None         
GoogLeNet    |  ProxyNCA/None      


__Online Products__

Architecture | Loss/Sampling |   NMI  |  F1  | Recall @ 1,10,100,1000
-------------|---------------|--------|------|-----------------
ResNet50     |               |        |      |     
GoogLeNet    |               |        |      |      



__In-Shop Clothes__

Architecture | Loss/Sampling       |   NMI  |  F1  | Recall @ 1,10,20,30,50
-------------|---------------      |--------|------|-----------------
ResNet50     |  Margin/Distance    | 88.2   | 27.7 | __84.5__, 96.1, 97.4, 97.9, 98.5
ResNet50     |  Triplet/Semihard   | __89.0__   | __30.8__ | 83.8, __96.4, 97.6, 98.2, 98.7__
ResNet50     |  NPair/None         | 88.0   | 27.6 | 80.9, 95.0, 96.6, 97.5, 98.2
 -           |         -           |    -   |   -  |  -   
GoogLeNet    |  Margin/Distance    | 86.9   | 23.0 | __78.9__, 91.8, 94.2, 95.3, 96.5
GoogLeNet    |  Triplet/Semihard   | 86.2   | 22.3 | 71.5, 90.2, 93.2, 94.5, 95.9
GoogLeNet    |  NPair/None         | __87.3__   | __25.3__ | 75.7, __92.6, 95.1, 96.2, 97.2__




---


## ToDO:
- [x] Fix Version in `requirements.txt`  
- [ ] Add Results for Implementations
- [x] Finalize Comments  
- [ ] Add Inception-BN  
- [ ] Add Lifted Structure Loss
