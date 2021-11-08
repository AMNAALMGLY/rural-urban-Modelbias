# Rural-Urban-Bias
Investigation of the sustainability models performance between rural and urban areas and deal with  the  loacation noise  in the data .

# Setting a baseline 
from Yeh, C., Perez, A., Driscoll, A. et al. Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. Nat Commun 11, 2583 (2020). https://doi.org/10.1038/s41467-020-16185-w


Till now this an implementaion of the dataset class , the model , and the training loop.More experiments need to be done.
to run this you have to configure directory of data in train.py then create environment and finally run train.py as follows:

````
conda env create --file i-mix.yml
````
````
python -m src.train.py
````
any new experiment you can add it to train.py.

# TODO
Revise anything and do a test module for every part 

add assertions for sanity checks

typing , documentation , citation 

threads /distributed learning

Results dataframe ['r2', 'R2', 'mse', 'rank'] evaluated_df function

# Reproducibility Experiments :
    knn NL mean scalar
    resnet18 MS
    resnet18 ms+NL concat (trained on NL seperately, ms seperately concat the fc , run ridge on top)
    transfer:resnet 18 RGB transfer


#New_experiments:
Drop dmsp nl for year <2012

concat features from images , with features learned with MLP for rural/urban column along with labels & locs columns

add cluster column(kmean) and then repeat the same experiment above .

