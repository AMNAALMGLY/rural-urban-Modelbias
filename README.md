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
add assertions for sanity checks

optimize dataloader class


feature extraction test

typing , documentation , citation 

threads /distributed learning

initialization function:ckpt , random , imagenet

Results dataframe ['r2', 'R2', 'mse', 'rank'] many metrics

Experimets :another models[knn,ridge], bands combination ,hyperparmeters

