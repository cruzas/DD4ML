## Note

To set up a test, you can create a configuration file to be run by run_config_file.py.

For example, you can type the following:
```bash
python3 run_config_file.py --sweep_config="./config_files/config_sgd.yaml"
```

It is not necessary to use wandb. You simply need to specify your logging function(s), which can be set up via callback functions, registered to a Trainer object: see ```src/trainer.py```. An example is given in ```run_config_file.py```.

## Sweeps
If you wish to perform a sweep with wandb, you can specify more than one value in the "values" field for each hyperparameter.

## Averaging 
In the command-line arguments, you can pass the number of trials that you wish to execute as ```--trials=num_trials```. 

After computing multiple trials, you can run ```compute_averages.py```, which will compute the best learning rate with respect to the average loss and accuracy for each batch size, based on the hyperparameter combinations seen.

## Configuration files
wandb configuration files are quite straightforward: ```config_sgd.yaml``` shows a small example.
If you find something that is not supported, such as a data set or network model, this can easily be added in ```utils.py``` in the global mapping directories at the top of the file. Of course, this will require you to create a dataset in ```src/datasets``` and a model in ```src/models```. 

## PMW
If you wish to use the Parallelized Model Wrapper (PMW) library that we developed, you can simply set ```--use_pmw=True```, but this will require the model to have an ```as_model_dict()``` function defined.


