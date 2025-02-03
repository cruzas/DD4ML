## Note

This is the tests folder. You can create a new folder for each test you wish to execute. 
Or, better yet, you can set up wandb configuration files in the ```tests/wandb/``` directory and run ```run_config_file.py```.

It is not necessary to use wandb, but it's currently the only option that will automatically save your results. Otherwise, what you log can be set up via callback functions. For example, see tests/wandb/run_config_file.py. Callback functions can be registered to a Trainer object: see src/trainer.py.

## Sweeps
If you wish to perform a sweep with wandb, you can specify more than one value in the "values" field for each hyperparameter.

## Averaging 
In the command-line arguments, you can pass the number of trials that you wish to execute as --trials=num_trials. 
See utils.parse_cmd_args() for the default values.

After computing multiple trials, you can run tests/wandb/compute_averages.py, which will compute the best learning rate with respect to the average loss and accuracy for each batch size, based on the hyperparameter combinations seen.

## Configuration files
wandb configuration files are quite straightforward: config_sgd.yaml shows a small example.
If you find something that is not supported, such as a data set or network model, this can easily be added in utils.py in the get_config_model_and_trainer() function. Of course, this will require you to create a dataset in src/datasets and a model in src/models. 

## PMW
If you wish to use the Parallelized Model Wrapper (PMW) library that we developed, you can simply set --use_pmw=True.


