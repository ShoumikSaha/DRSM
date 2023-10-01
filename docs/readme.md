# DRSM (De-Randomized Smoothed MalConv)

We redesigned the *de-randomized smoothing* scheme to make it applicable for byte-based inputs. We used MalConv as our base classifier and hence, we named our model **DRSM (De-Randomized Smoothed MalConv)**. We are also publishing our **PACE (Publicly Accessible Collection(s) of Executables)** dataset containing diverse benign raw executables.

Please cite our paper if you use our model or data.


## Setup
**Using Conda**

Simply run the command - 

`conda env create -n pytorch_env_smksaha --file environment.yml`

That's it! You're good to go! :smile:

## Training

To train the models, run `train_custom_malconv_by_ablation_from_csv.py`. Or, simply run the provided scripts, such as `train_script1.sh`, `train_script4.sh`, etc. 

Here, the *number* in the script name indicates the *n* in our DRSM-n models. For example, running `train_script4.sh` script would train our DRSM-4.

Below are the arguments for training - 

`--root_dir`: path of root directory that contains the dataset

`--train_path`: path of csv that contains the file paths from train-set

`--val_path`: path of csv that contains the file paths from validation-set

`--test_path`: path of csv that contains the file paths from test-set

`--dir_path`: directory where the trained model will be saved

`--ablation_idx`: the specific ablation of the model will be trained

`--ablations`: total number of ablations a model can have

`--dataset_size`: if you want to limit your train on specific number of samples. Otherwise, it will train on all files from 'train.csv'

`--epochs`: number of epochs the model will be trained

`--batch_size`: batch size to train on

`--non_neg`: True if you want to put non-negative weight constraint on the model


**We would recommend using our provided scripts to train the models. It's easier. You just need to provide root_directory, file_paths, and batch_size in that case.** After running this, the models will be saved in the `secml_malware/data/trained` folder. 

## Evaluation
### Standard Accuracy
Run the `evaluate_custom_malconv_by_ablation_from_csv.py` file. It takes almost the same arguments as training. However, you can simply run the `evaluate_script.sh` with just a few arguments.

After running this, it will output the standard accuracy with other metrics like false positive, true positive, confusion matrix etc.


### Certified Accuracy
Run the `evaluate_custom_malconv_cert_acc.py` file, or the script `cert_acc_script.sh`. It takes a list with `--perturb_size` argument for which it will generate the results.

## PACE Dataset
This dataset contains over 15.5K benign raw executables collected from different free websites. See our paper for a specific breakdown.

In the `dataset` folder, there are `.csv` files with website names, for example, `sourceforge.csv`. These csv files contain website URLs to download the executables from along with their MD5 hash. We will continue updating this dataset and extend it. 

| Version | Total Files |
|--------- |------------- |
| v1       | 15.5K |
| [v2](../dataset)       | 18.5K |



