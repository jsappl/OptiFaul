# Installation
This has been tested with
* Ubuntu 20.04.1 LTS (Linux 5.4.0-45-generic)
* Python 3.8.2
* CUDA 10.2

Clone this repo, initialize a virtual environment, and install all the necessary dependencies with
```shell
virtualenv -p python3 env
. env/bin/activate
pip install -r requirements.txt
```

# Training/testing the model
In the `src/` folder you can find [PyTorch](https://pytorch.org/) code for the machine learning part. We rely on the [PyTorch Forecasting](https://github.com/jdb78/pytorch-forecasting) library. You can adjust the settings to your liking and then run the whole code with the main runfile `run.py`.
```shell
nohup python3 run.py &
```
The data is preprocessed, the temporal fusion transformer is trained and tested, and some benchmark models are evaluated on the test data set. We use [TensorBoard](https://www.tensorflow.org/tensorboard/) to log hyperparameters, train/validation loss, and the performance of the model on the test data set. Run
```shell
tensorboard --logdir runs/ &
```
if you want to monitor the loss during training and check the test results in your browser.

We have provided the training results of one of our runs [here](https://tensorboard.dev/experiment/3kVF0zqmSpGbgPxteeWjDQ/).
