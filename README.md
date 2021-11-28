# Rural-Urban-Bias
Investigation of the sustainability models performance between rural and urban areas and deal with  the  loacation noise  in the data .

# Setting a baseline 
from Yeh, C., Perez, A., Driscoll, A. et al. Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. Nat Commun 11, 2583 (2020). https://doi.org/10.1038/s41467-020-16185-w


To run this you have to configure directory of data in train.py then create environment and finally run train.py as follows:

````
conda env create --file envi.yml
````
````
python -m src.train2
````
any new experiment you can add it to train2.py.


# New_experiments:

add open building dataset

add covertype dataset 

Drop dmsp nl for year <2012

add cluster column(kmean) and then repeat the same experiments above .

# Sanity checks Experiments :
knn NL mean scalar

resnet18 MS

resnet18 ms+NL concat (trained on NL seperately, ms seperately concat the fc , run ridge on top)

transfer:resnet 18 RGB transfer

# TODO

add assertions for sanity checks

typing , documentation , citation 

threads /distributed learning
