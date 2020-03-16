# Easily Extendable Basic Deep Metric Learning Pipeline
#### Karsten Roth (karsten.rh1@gmail.com), Biagio Brattoli (biagio.brattoli@gmail.com)

*When using this repo in any academic work, please provide a reference to this repo, that would be greatly appreciated :).*

---

#### Based on an extendend version of this repo, we have created a thorough comparison and evaluation of Deep Metric Learning: 
https://arxiv.org/abs/2002.08473

The used code will be uploaded soon :).

---
#### For usage, go to section 3 - for results to section 4

## 1. Overview
This repository contains a full, easily extendable pipeline to test and implement current and new deep metric learning methods. For referencing and testing, this repo contains implementations/dataloaders for:

__Loss Functions__
* Triplet Loss (https://arxiv.org/abs/1412.6622)
* Margin Loss (https://arxiv.org/abs/1706.07567)
* ProxyNCA (https://arxiv.org/abs/1703.07464)
* N-Pair Loss (https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)

__Sampling Methods__
* Random Sampling
* Softhard Sampling (Soft Version of hard tuple sampling)
* Semihard Sampling (https://arxiv.org/abs/1503.03832)
* Distance Sampling (https://arxiv.org/abs/1706.07567)
* N-Pair Sampling (https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)

__Datasets__
* CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)
* CARS196 (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)
* In-Shop Clothes (http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html, download from https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E. Thanks to KunHe for providing the link!)
* (optional) PKU Vehicle-ID (https://www.pkuml.org/resources/pku-vds.html)

__Architectures__
* ResNet50 (https://arxiv.org/abs/1512.03385)
* GoogLeNet (https://arxiv.org/abs/1409.4842)    
  [Note: This version follows the official torchvision implementation, which differs from the original variant.]


__NOTE__: PKU Vehicle-ID is _(optional)_ because there is no direct way to download the dataset, as it requires special licensing. However, if this dataset becomes available (in the structure shown in part 2.2), it can be used directly.

---
### 1.1 Related Repos:
* [Metric Learning with Mined Interclass Characteristics](https://github.com/Confusezius/metric-learning-mining-interclass-characteristics)
* [Metric Learning by dividing the embedding space](https://github.com/CompVis/metric-learning-divide-and-conquer)
* [Deep Metric Learning to Rank](https://github.com/kunhe/FastAP-metric-learning)
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
|   Standard_Training.py     (main training script)
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
__CUB200-2011/CARS196__
```
cub200/cars196
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
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
└─img
|    └─MEN
|         └─Denim
|               └─id_00000080
|                  │   01_1_front.jpg
|                  │   ...
|               ...
|         ...
|    ...
|
└─Eval
|  │   list_eval_partition.txt
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
**NOTE:** If one wishes not to use `faiss` but standard `sklearn`, simply use `auxiliaries_nofaiss.py` to replace `auxiliaries.py` when importing the libraries.



### [2.] Exemplary Runs
The main script is `Standard_Training.py`. If running without input arguments, training of ResNet50 on CUB200-2011 with Marginloss and Distance-sampling is performed.  
Otherwise, the following flags suffice to train with different losses, sampling methods, architectures and datasets:
```
python Standard_Training.py --dataset <dataset> --loss <loss> --sampling <sampling> --arch <arch> --k_vals <k_vals> --embed_dim <embed_dim>
```
The following flags are available:
* `<dataset> <- cub200, cars196, online_products, in-shop, vehicle_id`
* `<loss> <- marginloss, triplet, npair, proxynca`
* `<sampling> <- distance, semihard, random, npair`
* `<arch> <- resnet50, googlenet`
* `<k_vals> <- List of Recall @ k values to evaluate on, e.g. 1 2 4 8`
* `<embed_dim> <- Network embedding dimension. Default: 128 for ResNet50, 512 for GoogLeNet.`

For all other training-specific arguments (e.g. batch-size, num. training epochs., ...), simply refer to the input arguments in `Standard_Training.py`.

__NOTE__: If one wishes to use a different learning rate for the final linear embedding layer, the flag `--fc_lr_mul` needs to be set to a value other than zero (i.e. `10` as is done in various implementations).

Finally, to decide the GPU to use and the name of the training folder in which network weights, sample recoveries and metrics are stored, set:
```
python Standard_Training.py --gpu <gpu_id> --savename <name_of_training_run>
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


### [4.] Stored Data:
By default, the following files are saved:
```
Name_of_Training_Run
|  checkpoint.pth.tar   -> Contains network state-dict.
|  hypa.pkl             -> Contains all network parameters as pickle.
|                          Can be used directly to recreate the network.
| log_train_Base.csv    -> Logged training data as CSV.                      
| log_val_Base.csv      -> Logged test metrics as CSV.                    
| Parameter_Info.txt    -> All Parameters stored as readable text-file.
| InfoPlot_Base.svg     -> Graphical summary of training/testing metrics progression.
| sample_recoveries.png -> Sample recoveries for best validation weights.
|                          Acts as a sanity test.
```

![Sample Recoveries](/Images/sample_recoveries.png)
__Note:__ _Red denotes query images, while green show the resp. nearest neighbours._

![Sample Recoveries](/Images/InfoPlot_Base.png)
__Note:__ _The header in the summary plot shows the best testing metrics over the whole run._

### [5.] Additional Notes:
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
These results are supposed to be performance estimates achieved by running the respective commands in `sample_training_runs.sh`. Note that the learning rate scheduling might not be fully optimised, so these values should only serve as reference/expectation, not what can be ultimately achieved with more tweaking.

_Note also that there is a not insignificant dependency on the used seed._



__CUB200__

Architecture | Loss/Sampling       |   NMI  |  F1  | Recall @ 1 -- 2 -- 4 -- 8
-------------|---------------      |--------|------|-----------------
ResNet50     |  Margin/Distance    | __68.2__   | __38.7__ | 63.4 -- 74.9 --  __86.0__ --  90.4    
ResNet50     |  Triplet/Semihard   | 66.2   | 35.5 | 61.2 --  73.2 --  82.4 --  89.5    
ResNet50     |  NPair/None         | 65.4   | 33.8 | 59.0 --  71.3 --  81.1 --  88.8    
ResNet50     |  ProxyNCA/None      | 68.1   | 38.1 | __64.0__ --  __75.4__ --  84.2 --  __90.5__    


__Cars196__

Architecture | Loss/Sampling       |   NMI  |  F1  | Recall @ 1 -- 2 -- 4 -- 8
-------------|---------------      |--------|------|-----------------
ResNet50     |  Margin/Distance    | __67.2__   | __37.6__ | 79.3 -- 87.1 -- __92.1 -- 95.4__    
ResNet50     |  Triplet/Semihard   | 64.4   | 32.4 | 75.4 -- 84.2 -- 90.1 -- 94.1
ResNet50     |  NPair/None         | 62.3   | 30.1 | 69.5 -- 80.2 -- 87.3 -- 92.1
ResNet50     |  ProxyNCA/None      | 66.3   | 35.8 | __80.0 -- 87.2__ -- 91.8 -- 95.1

__Online Products__

Architecture | Loss/Sampling       |   NMI  |  F1  | Recall @ 1 -- 10 -- 100 -- 1000
-------------|---------------      |--------|------|-----------------
ResNet50     |  Margin/Distance    | __89.6__   | __34.9__ | __76.1 -- 88.7 -- 95.1__ -- 98.3
ResNet50     |  Triplet/Semihard   | 89.1   | 33.7 | 74.3 -- 87.6 -- 94.9 -- __98.5__
ResNet50     |  NPair/None         | 88.8   | 31.1 | 70.9 -- 85.2 -- 93.8 -- 98.2


__In-Shop Clothes__

Architecture | Loss/Sampling       |   NMI  |  F1  | Recall @ 1 -- 10 -- 20 -- 30 -- 50
-------------|---------------      |--------|------|-----------------
ResNet50     |  Margin/Distance    | 88.2   | 27.7 | __84.5__ -- 96.1 -- 97.4 -- 97.9 -- 98.5
ResNet50     |  Triplet/Semihard   | __89.0__   | __30.8__ | 83.9 -- __96.3 -- 97.6 -- 98.4 -- 98.8__
ResNet50     |  NPair/None         | 88.0   | 27.6 | 80.9 -- 95.0 -- 96.6 -- 97.5 -- 98.2


__NOTE:__
 1. Regarding __Vehicle-ID__: Due to the number of test sets, size of the training set and little public accessibility, results are not included for the time being.
 2. Regarding ProxyNCA for __Online Products__ and __In-Shop Clothes__: Due to the high number of classes, the number of proxies required is too high for useful training (>10000 proxies).

---

## ToDO:
- [x] Fix Version in `requirements.txt`  
- [x] Add Results for Implementations
- [x] Finalize Comments  
- [ ] Add Inception-BN  
- [ ] Add Lifted Structure Loss
