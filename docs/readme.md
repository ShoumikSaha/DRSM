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

## Evaluation

## Dataset
