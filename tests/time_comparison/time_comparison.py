import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.datasets.char_dataset import CharDataset
from src.models.gpt.mingpt.model import GPT as SequentialGPT
from src.models.gpt.mingpt.trainer import Trainer
from src.models.gpt.skgpt.model import GPT as ParallelGPT
from src.pmw.base_model import BaseModel
from src.pmw.model_handler import ModelHandler
from src.pmw.parallelized_model import ParallelizedModel
from src.utils import CfgNode as CN
from src.utils import dprint, set_seed


def get_config(GPT):

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = '../../saved_networks/chargpt/'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    # the model we're using is so small that we can go a bit faster
    C.trainer.learning_rate = 5e-4

    return C

def measure_time(model, trainer_class, dataset, sample_input, sample_target, trainer_config=None, is_parallel=False):
    if is_parallel:
        # Prepare parallelized setup
        model_handler = ModelHandler(model.model_dict, num_subdomains=1, num_replicas_per_subdomain=1)
        model = ParallelizedModel(model_handler, sample=sample_input)

    trainer = trainer_class(trainer_config, model, dataset)

    # Measure forward and backward pass times
    inputs = sample_input.to(trainer.device)
    targets = sample_target.to(trainer.device)

    forward_times = []
    backward_times = []
    num_iterations = 10

    for _ in range(num_iterations):
        # Forward pass
        start_time = time.time()
        outputs = model(inputs)[0]
        forward_times.append(time.time() - start_time)

        # Reshape outputs and targets for loss computation
        outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * block_size, vocab_size)
        targets = targets.view(-1)  # (batch_size * block_size)

        # Backward pass
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets)
        start_time = time.time()
        loss.backward()
        backward_times.append(time.time() - start_time)

    # Compute average times
    avg_forward_time = sum(forward_times) / num_iterations
    avg_backward_time = sum(backward_times) / num_iterations

    return avg_forward_time, avg_backward_time


def main(rank=None, world_size=None, args=None):
    # Initialize the process group
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",  # Or another free port
        rank=rank,
        world_size=world_size
    )
    
    config = get_config(SequentialGPT)
    set_seed(config.system.seed)  # Ensure reproducibility

    text = open('../../input.txt', 'r').read()
    dataset = CharDataset(config.data, text)
    sample_input = dataset.get_sample_input(config.trainer)
    sample_target = dataset.get_sample_target(config.trainer)

    if rank == 0:
        # Sequential model setup
        seq_config = config
        seq_config.model.vocab_size = dataset.get_vocab_size()
        seq_config.model.block_size = dataset.get_block_size()
        sequential_model = SequentialGPT(seq_config.model)
        print(f"Sequential config: {config}")

        seq_forward_time, seq_backward_time = measure_time(
            sequential_model, Trainer, dataset, sample_input, sample_target, seq_config.trainer
        )

        print("Sequential Model:")
        print(f"Forward pass time: {seq_forward_time:.4f} seconds")
        print(f"Backward pass time: {seq_backward_time:.4f} seconds")

    # Parallel model setup
    par_config = get_config(ParallelGPT)
    par_config.merge_from_dict(args)
    par_config.merge_and_cleanup()
    par_config.model.vocab_size = dataset.get_vocab_size()
    par_config.model.block_size = dataset.get_block_size()
    par_config.model.num_stages = 2
    parallel_model = ParallelGPT(par_config.model)
    dprint(f"Parallel config: {par_config}")
    BaseModel.n_layer = par_config.model.n_layer # Bit of a hack for now

    par_forward_time, par_backward_time = measure_time(
        parallel_model, Trainer, dataset, sample_input, sample_target, par_config.trainer, is_parallel=True
    )

    print("\nParallel Model:")
    print(f"Forward pass time: {par_forward_time:.4f} seconds")
    print(f"Backward pass time: {par_backward_time:.4f} seconds")


if __name__ == '__main__':
    num_subdomains = 1
    num_replicas_per_subdomain = 1
    num_stages = 2
    num_shards = 1

    args_dict = {
        'num_subdomains': num_subdomains,
        'num_replicas_per_subdomain': num_replicas_per_subdomain,
        'num_stages': num_stages,
        'num_shards': num_shards
    }

    world_size = num_subdomains * num_replicas_per_subdomain * num_stages * num_shards
    mp.spawn(
        main,
        args=(world_size, args_dict),
        nprocs=world_size,
        join=True
    )