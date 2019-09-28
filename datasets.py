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

################# LIBRARIES ###############################
import warnings
warnings.filterwarnings("ignore")

import numpy as np, os, sys, pandas as pd, csv, copy
import torch, torch.nn as nn, matplotlib.pyplot as plt, random

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import pretrainedmodels.utils as utils
import auxiliaries as aux


"""============================================================================"""
################ FUNCTION TO RETURN ALL DATALOADERS NECESSARY ####################
def give_dataloaders(dataset, opt):
    """
    Args:
        dataset: string, name of dataset for which the dataloaders should be returned.
        opt:     argparse.Namespace, contains all training-specific parameters.
    Returns:
        dataloaders: dict of dataloaders for training, testing and evaluation on training.
    """
    #Dataset selection
    if opt.dataset=='cub200':
        datasets = give_CUB200_datasets(opt)
    elif opt.dataset=='cars196':
        datasets = give_CARS196_datasets(opt)
    elif opt.dataset=='online_products':
        datasets = give_OnlineProducts_datasets(opt)
    elif opt.dataset=='in-shop':
        datasets = give_InShop_datasets(opt)
    elif opt.dataset=='vehicle_id':
        datasets = give_VehicleID_datasets(opt)
    else:
        raise Exception('No Dataset >{}< available!'.format(dataset))

    #Move datasets to dataloaders.
    dataloaders = {}
    for key,dataset in datasets.items():
        is_val = dataset.is_validation
        dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.kernels, shuffle=not is_val, pin_memory=True, drop_last=not is_val)

    return dataloaders


"""============================================================================"""
################# FUNCTIONS TO RETURN TRAIN/VAL PYTORCH DATASETS FOR CUB200, CARS196, STANFORD ONLINE PRODUCTS, IN-SHOP CLOTHES, PKU VEHICLE-ID ####################################
def give_CUB200_datasets(opt):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the CUB-200-2011 dataset.
    For Metric Learning, the dataset classes are sorted by name, and the first half used for training while the last half is used for testing.
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    image_sourcepath  = opt.source_path+'/images'
    #Find available data classes.
    image_classes = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0]))
    #Make a index-to-labelname conversion dict.
    conversion    = {int(x.split('.')[0]):x.split('.')[-1] for x in image_classes}
    #Generate a list of tuples (class_label, image_path)
    image_list    = {int(key.split('.')[0]):sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    #Image-dict of shape {class_idx:[list of paths to images belong to this class] ...}
    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    keys = sorted(list(image_dict.keys()))

    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]
    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}


    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    val_dataset   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt, is_validation=True)

    train_dataset.conversion = conversion
    val_dataset.conversion   = conversion
    eval_dataset.conversion  = conversion

    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}


def give_CARS196_datasets(opt):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the CARS196 dataset.
    For Metric Learning, the dataset classes are sorted by name, and the first half used for training while the last half is used for testing.
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    image_sourcepath  = opt.source_path+'/images'
    #Find available data classes.
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    #Make a index-to-labelname conversion dict.
    conversion    = {i:x for i,x in enumerate(image_classes)}
    #Generate a list of tuples (class_label, image_path)
    image_list    = {i:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key)]) for i,key in enumerate(image_classes)}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    #Image-dict of shape {class_idx:[list of paths to images belong to this class] ...}
    image_dict    = {}
    for key, img_path in image_list:
        key = key
        # key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    keys = sorted(list(image_dict.keys()))

    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]
    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}

    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    val_dataset   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt, is_validation=True)

    train_dataset.conversion = conversion
    val_dataset.conversion   = conversion
    eval_dataset.conversion  = conversion

    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}


def give_OnlineProducts_datasets(opt):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the Online-Products dataset.
    For Metric Learning, training and test sets are provided by given text-files, Ebay_train.txt & Ebay_test.txt.
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    image_sourcepath  = opt.source_path+'/images'
    #Load text-files containing classes and imagepaths.
    training_files = pd.read_table(opt.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(opt.source_path+'/Info_Files/Ebay_test.txt', header=0, delimiter=' ')

    #Generate Conversion dict.
    conversion = {}
    for class_id, path in zip(training_files['class_id'],training_files['path']):
        conversion[class_id] = path.split('/')[0]
    for class_id, path in zip(test_files['class_id'],test_files['path']):
        conversion[class_id] = path.split('/')[0]

    #Generate image_dicts of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict, val_image_dict  = {},{}
    for key, img_path in zip(training_files['class_id'],training_files['path']):
        key = key-1
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(test_files['class_id'],test_files['path']):
        key = key-1
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_image_dict[key].append(image_sourcepath+'/'+img_path)

    ### Uncomment this if super-labels should be used to generate resp.datasets
    # super_conversion = {}
    # for super_class_id, path in zip(training_files['super_class_id'],training_files['path']):
    #     conversion[super_class_id] = path.split('/')[0]
    # for key, img_path in zip(training_files['super_class_id'],training_files['path']):
    #     key = key-1
    #     if not key in super_train_image_dict.keys():
    #         super_train_image_dict[key] = []
    #     super_train_image_dict[key].append(image_sourcepath+'/'+img_path)
    # super_train_dataset = BaseTripletDataset(super_train_image_dict, opt, is_validation=True)
    # super_train_dataset.conversion = super_conversion


    train_dataset       = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    val_dataset         = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
    eval_dataset        = BaseTripletDataset(train_image_dict, opt, is_validation=True)

    train_dataset.conversion       = conversion
    val_dataset.conversion         = conversion
    eval_dataset.conversion        = conversion

    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}
    # return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset, 'super_evaluation':super_train_dataset}


def give_InShop_datasets(opt):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the In-Shop Clothes dataset.
    For Metric Learning, training and test sets are provided by one text file, list_eval_partition.txt.
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing (by query and gallery separation) and evaluation.
    """
    #Load train-test-partition text file.
    data_info = np.array(pd.read_table(opt.source_path+'/Eval/list_eval_partition.txt', header=1, delim_whitespace=True))[1:,:]
    #Separate into training dataset and query/gallery dataset for testing.
    train, query, gallery   = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]

    #Generate conversions
    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
    train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])

    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
    query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
    gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

    #Generate Image-Dicts for training, query and gallery of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict    = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(opt.source_path+'/'+img_path)

    query_image_dict    = {}
    for img_path, key in query:
        if not key in query_image_dict.keys():
            query_image_dict[key] = []
        query_image_dict[key].append(opt.source_path+'/'+img_path)

    gallery_image_dict    = {}
    for img_path, key in gallery:
        if not key in gallery_image_dict.keys():
            gallery_image_dict[key] = []
        gallery_image_dict[key].append(opt.source_path+'/'+img_path)

    ### Uncomment this if super-labels should be used to generate resp.datasets
    # super_train_image_dict, counter, super_assign = {},0,{}
    # for img_path, _ in train:
    #     key = '_'.join(img_path.split('/')[1:3])
    #     if key not in super_assign.keys():
    #         super_assign[key] = counter
    #         counter += 1
    #     key = super_assign[key]
    #
    #     if not key in super_train_image_dict.keys():
    #         super_train_image_dict[key] = []
    #     super_train_image_dict[key].append(opt.source_path+'/'+img_path)
    # super_train_dataset = BaseTripletDataset(super_train_image_dict, opt, is_validation=True)

    train_dataset     = BaseTripletDataset(train_image_dict, opt,   samples_per_class=opt.samples_per_class)
    eval_dataset      = BaseTripletDataset(train_image_dict, opt,   is_validation=True)
    query_dataset     = BaseTripletDataset(query_image_dict, opt,   is_validation=True)
    gallery_dataset   = BaseTripletDataset(gallery_image_dict, opt, is_validation=True)

    return {'training':train_dataset, 'testing_query':query_dataset, 'evaluation':eval_dataset, 'testing_gallery':gallery_dataset}
    # return {'training':train_dataset, 'testing_query':query_dataset, 'evaluation':eval_dataset, 'testing_gallery':gallery_dataset, 'super_evaluation':super_train_dataset}


def give_VehicleID_datasets(opt):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the PKU Vehicle dataset.
    For Metric Learning, training and (multiple) test sets are provided by separate text files, train_list and test_list_<n_classes_2_test>.txt.
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    #Load respective text-files
    train       = np.array(pd.read_table(opt.source_path+'/train_test_split/train_list.txt', header=None, delim_whitespace=True))
    small_test  = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_800.txt', header=None, delim_whitespace=True))
    medium_test = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_1600.txt', header=None, delim_whitespace=True))
    big_test    = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_2400.txt', header=None, delim_whitespace=True))

    #Generate conversions
    lab_conv = {x:i for i,x in enumerate(np.unique(train[:,1]))}
    train[:,1] = np.array([lab_conv[x] for x in train[:,1]])
    lab_conv = {x:i for i,x in enumerate(np.unique(np.concatenate([small_test[:,1], medium_test[:,1], big_test[:,1]])))}
    small_test[:,1]  = np.array([lab_conv[x] for x in small_test[:,1]])
    medium_test[:,1] = np.array([lab_conv[x] for x in medium_test[:,1]])
    big_test[:,1]    = np.array([lab_conv[x] for x in big_test[:,1]])

    #Generate Image-Dicts for training and different testings of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict    = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    small_test_dict = {}
    for img_path, key in small_test:
        if not key in small_test_dict.keys():
            small_test_dict[key] = []
        small_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    medium_test_dict    = {}
    for img_path, key in medium_test:
        if not key in medium_test_dict.keys():
            medium_test_dict[key] = []
        medium_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    big_test_dict    = {}
    for img_path, key in big_test:
        if not key in big_test_dict.keys():
            big_test_dict[key] = []
        big_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt,    is_validation=True)
    val_small_dataset     = BaseTripletDataset(small_test_dict, opt,  is_validation=True)
    val_medium_dataset    = BaseTripletDataset(medium_test_dict, opt, is_validation=True)
    val_big_dataset       = BaseTripletDataset(big_test_dict, opt,    is_validation=True)

    return {'training':train_dataset, 'testing_set1':val_small_dataset, 'testing_set2':val_medium_dataset, \
            'testing_set3':val_big_dataset, 'evaluation':eval_dataset}





################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseTripletDataset(Dataset):
    """
    Dataset class to provide (augmented) correctly prepared training samples corresponding to standard DML literature.
    This includes normalizing to ImageNet-standards, and Random & Resized cropping of shapes 224 for ResNet50 and 227 for
    GoogLeNet during Training. During validation, only resizing to 256 or center cropping to 224/227 is performed.
    """
    def __init__(self, image_dict, opt, samples_per_class=8, is_validation=False):
        """
        Dataset Init-Function.

        Args:
            image_dict:         dict, Dictionary of shape {class_idx:[list of paths to images belong to this class] ...} providing all the training paths and classes.
            opt:                argparse.Namespace, contains all training-specific parameters.
            samples_per_class:  Number of samples to draw from one class before moving to the next when filling the batch.
            is_validation:      If is true, dataset properties for validation/testing are used instead of ones for training.
        Returns:
            Nothing!
        """
        #Define length of dataset
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])

        self.is_validation = is_validation

        self.pars        = opt
        self.image_dict  = image_dict

        self.avail_classes    = sorted(list(self.image_dict.keys()))

        #Convert image dictionary from classname:content to class_idx:content, because the initial indices are not necessarily from 0 - <n_classes>.
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))

        #Init. properties that are used when filling up batches.
        if not self.is_validation:
            self.samples_per_class = samples_per_class
            #Select current class to sample images from up to <samples_per_class>
            self.current_class   = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0

        #Data augmentation/processing methods.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf_list = []
        if not self.is_validation:
            transf_list.extend([transforms.RandomResizedCrop(size=224) if opt.arch=='resnet50' else transforms.RandomResizedCrop(size=227),
                                transforms.RandomHorizontalFlip(0.5)])
        else:
            transf_list.extend([transforms.Resize(256),
                                transforms.CenterCrop(224) if opt.arch=='resnet50' else transforms.CenterCrop(227)])

        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)

        #Convert Image-Dict to list of (image_path, image_class). Allows for easier direct sampling.
        self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        #Flag that denotes if dataset is called for the first time.
        self.is_init = True


    def ensure_3dim(self, img):
        """
        Function that ensures that the input img is three-dimensional.

        Args:
            img: PIL.Image, image which is to be checked for three-dimensionality (i.e. if some images are black-and-white in an otherwise coloured dataset).
        Returns:
            Checked PIL.Image img.
        """
        if len(img.size)==2:
            img = img.convert('RGB')
        return img


    def __getitem__(self, idx):
        """
        Args:
            idx: Sample idx for training sample
        Returns:
            tuple of form (sample_class, torch.Tensor() of input image)
        """
        if self.is_init:
            self.current_class = self.avail_classes[idx%len(self.avail_classes)]
            self.is_init = False

        if not self.is_validation:
            if self.samples_per_class==1:
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

            if self.n_samples_drawn==self.samples_per_class:
                #Once enough samples per class have been drawn, we choose another class to draw samples from.
                #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                #previously or one before that.
                counter = copy.deepcopy(self.avail_classes)
                for prev_class in self.classes_visited:
                    if prev_class in counter: counter.remove(prev_class)

                self.current_class   = counter[idx%len(counter)]
                self.classes_visited = self.classes_visited[1:]+[self.current_class]
                self.n_samples_drawn = 0

            class_sample_idx = idx%len(self.image_dict[self.current_class])
            self.n_samples_drawn += 1

            out_img = self.transform(self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
            return self.current_class,out_img
        else:
            return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

    def __len__(self):
        return self.n_files
