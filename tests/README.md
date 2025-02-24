## Usage

It is relatively straightforward to execute a test.

In a ***local*** environment (e.g. PC), you can, for example, run the following:
```bash
open -a Docker
wandb server start
python3 ./tests/run_config_file.py --sweep_config="./tests/config_files/config_sgd.yaml"
```

In a ***cluster*** environment, you need to ensure that, in ```submit_job.sh```, you set ```JOB_NAME``` to be the name of the Python file you wish to execute. If you intend to use wandb, set ```USE_WANDB=1``` in ```submit_job.sh```. Of course, change the time/number of GPUs/etc. as needed in ```job_template.job```.
