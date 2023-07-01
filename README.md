# Establishing-A-Generic-Methodology-For-Out-Of-Distribution-Image-Detection
This repository contains the models, data and code for project ["Establishing A Generic Methodology For Out-Of-Distribution Image Detection"](https://drive.google.com/file/d/1-Pv5LuaBYC9b8AfGU315kRaAFsPj1_jv/view?usp=sharing).

Before getting started, unzip [data.zip](https://drive.google.com/file/d/1oB0ARu4fVfIOMrhexbIimFIhuaTxXudm/view?usp=drive_link) and [models.zip](https://drive.google.com/file/d/1dBVHAF5j8XKl68QSRaf9-FyBu67Rp1rN/view?usp=drive_link) and place it in the storage folder.

Within the data folder, there is a "modelrunresult" folder. This folder holds the y scores for each data point when Test.py or Test100.py is ran. The scores are then used to compute the AUROC and AUPR. In the interest of keeping the results for comparison across different number of models in the ensemble, it is further divided into folders based on the number of models selected to form the ensemble.

Within the models > trees > cifar100 folder, the tree models are split into different folders based on the top N number of results taken for classifying in or out of distribution.

There are a few features provided in this repository.
1. Replicating the results in the project
Run either command (Test.py for CIFAR10 and Test100.py for CIFAR100)
`python Test.py`
`python Test100.py`
2. Finding the fixed softmax threshold for a particular true positive rate
`python FindFixedSoftmaxThreshold.py`
3. Build the trees from trained models to determine the threshold (BuildTrees.py for CIFAR10 and BuildTrees100.py for CIFAR100)
`python BuildTrees.py`
`python BuildTrees100.py`
There are also arguments available for each script to vary the parameters of the run. Refer to the script itself for more details.

The package and versions used are:
1. torch v1.11.0
2. torchvision v0.12.0
3. sklearn v1.2.2
4. pickle v4.0
5. numpy v1.22.2