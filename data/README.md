## Setting up the data

The benchmarking and demonstration scripts expects the dataset 
from https://www.kaggle.com/datasets/laotse/credit-risk-dataset to be in the `data/credit_risk_dataset.csv` dataset.

To setup the data for benchmarking + fairness analysis, download the data from that directory and use the `prepare_data.py` script to spit it into 4 batches for incremental learning simulation.


```shell
kaggle datasets download laotse/credit-risk-dataset && unzip credit-risk-dataset.zip
python prepare_data.py --num_batches 4 --bias_prob 0.65
```

The bias_prob flag above can be used to adjust the probability for adding the bias_variable according to the following generative model:

```
    If the loan is defaulted i.e. prediction class 1:
      assign bias_variable = 0 or 1 with the probability of 0 being {bias_prob}

    if the loan is not defaulted i.e. prediction class 0:
      assign bias_variable = 0 or 1 with the probability of 0 being {1 - bias_prob}
```

This allows for the user to investigate how the metrics may hold up to various levels of this introduced bias.

> **Please see this data set's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.**

### Installing Kaggle CLI

If the above kaggle command does not work, do the following

1) Install kaggle if not done using the below command:  
    ```bash
    pip install kaggle
    ```
2) Login to Kaggle account. Go to 'Account Tab' & select 'Create a new API token'. This will trigger the download of kaggle.json file. This file contains your API credentials.
3) Move the downloaded 'kaggle.json' file to the .kaggle folder in your home directory (~/.kaggle/)
4) Execute the following command:
    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```
5) Export the kaggle username & token to the enviroment, but you may not need this if the API token is placed in the right location
    ```bash
    export KAGGLE_USERNAME=@@@@@@@@$#$#$
    export KAGGLE_KEY=@#@#@#@#@#@#!!@@@@##########
    ```

1) Install kaggle if not done using the below command:  
    ```bash
    pip install kaggle
    ```
2) Login to Kaggle account. Go to 'Account Tab' & select 'Create a new API token'. This will trigger the download of kaggle.json file. This file contains your API credentials.
3) Move the downloaded 'kaggle.json' file to the .kaggle folder in your home directory (~/.kaggle/)
4) Execute the following command:
    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```
5) Export the kaggle username & token to the enviroment, but you may not need this if the API token is placed in the right location
    ```bash
    export KAGGLE_USERNAME=@@@@@@@@$#$#$
    export KAGGLE_KEY=@#@#@#@#@#@#!!@@@@##########
    ```