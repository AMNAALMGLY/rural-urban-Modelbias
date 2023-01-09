# Rural-Urban Model bias

This is the official implementation of the 2022 SIGSPATIAL paper `Understanding Economic Development using Satallite Images, Buildings footprints and Deep models.`

[Paper](https://dl.acm.org/doi/pdf/10.1145/3557915.3561025/)

[Video](https://drive.google.com/drive/folders/1igkCEP1TKNykVPYCcPiaW9ns3VkuLYJd?usp=sharing)
## Configuration

To understand different components/configs in the experiments, we recommend first looking at ```config.py``` where you can change different configs for dataset, model, and training.

## Dataset

The satellite images used aren't publicly available yet, but the ground truth is under  ```data/dhs_labels_quantile.csv```

## Models

In the experiments we tried different types of models, all present under `models` folder, however the best results are achieved using preact resnet+ self attention layers

## Training
To run training from scratch you need first to make sure to make changes in ```config.py``` for your data directory then run the following:

````
conda env create --file envi.yml
````
````
python -m src.train2
````

## Testing
In ```notebooks/dhs_ooc2.ipynb``` or ```notebooks/dhs_ooc-otherlabels.ipynb```  you can test different trained models and analyze the results

## Acnoweldgment
The code base of this project is based on this paper  `Using publicly available satellite imagery and deep learning to understand economic well-being in Africa` [Repository](https://github.com/chrisyeh96/africa_poverty_clean)

## Citation
you can cite this work as 
```
@inproceedings{elmustafa2022understanding,
  title={Understanding economic development in rural Africa using satellite imagery, building footprints and deep models},
  author={Elmustafa, Amna and Rozi, Erik and He, Yutong and Mai, Gengchen and Ermon, Stefano and Burke, Marshall and Lobell, David},
  booktitle={Proceedings of the 30th International Conference on Advances in Geographic Information Systems},
  pages={1--4},
  year={2022},
}

```
## TODO
* List table of results
* Add the satellite & buildings dataset
* More details  on Experiments and models
* Modre details on ground truth data



