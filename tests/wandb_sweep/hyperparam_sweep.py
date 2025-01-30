import os

import torch
import torch.multiprocessing as mp
import yaml

from src.utils import distributed_run, dprint, generic_run, parse_cmd_args

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def main(rank=None, master_addr=None, master_port=None, world_size=None, args=None, wandb_config=None, use_wandb=False):
    logging_fn = dprint if not use_wandb else (lambda x: wandb.log(x))

    def epoch_end_callback(trainer, save_model=True, save_frequency=1):
        dprint(f"Epoch {trainer.epoch_num}, Loss: {trainer.loss:.4f}, Accuracy: {trainer.accuracy:.2f}%, Time: {trainer.epoch_dt:.2f}s")
        if rank == 0 and use_wandb:
            logging_fn({
                "epoch": trainer.epoch_num,
                "epoch_time": trainer.epoch_dt,
                "loss": trainer.loss,
                "accuracy": trainer.accuracy,
                "running_time": trainer.running_time
            })

        if save_model and trainer.epoch_num % save_frequency == 0:
            dprint("Saving model...")
            model_path = os.path.join(args["work_dir"], f"model_epoch_{trainer.epoch_num}.pt")
            # Check if args["work_dir"] exists
            if not os.path.exists(args["work_dir"]):
                os.makedirs(args["work_dir"])
            
            if args["use_pmw"]:
                trainer.model.save_state_dict(model_path)
            else:
                torch.save(trainer.model.state_dict(), model_path)
    
    # batch_end_callback is not used in this example. It is more appropriate when training transformer models.
    # def batch_end_callback(trainer):
    #     dprint(f"Epoch [{trainer.epoch_num}/{trainer.config.epochs}] {trainer.epoch_progress}%\r")
    
    generic_run(rank=rank, master_addr=master_addr, master_port=master_port, world_size=world_size, args=args, wandb_config=wandb_config if use_wandb else None, epoch_end_callback=epoch_end_callback, batch_end_callback=None)
        
def one_trial_hyperparam_sweep(args):
    args_dict = vars(args)
    sweep_id = None  # Initialize once
    entity = None
    project = None
    
    if WANDB_AVAILABLE:
        os.environ["WANDB_START_METHOD"] = "thread"  
        with open(args.sweep_config, "r") as file:
            sweep_config = yaml.safe_load(file)
        
    entity = args.entity
    project = args.project
    num_workers = args.num_workers  # or however many local processes you want
    sweep_id = wandb.sweep(sweep_config, project=project)  # No need for redundant check
    distributed_run(main, num_workers, sweep_id, entity, project, args_dict, WANDB_AVAILABLE)
        
if __name__ == "__main__":
    args = parse_cmd_args()
    
    for trial in range(args.trials):
        dprint(f"Running trial {trial+1}/{args.trials}")
        one_trial_hyperparam_sweep(args)
    