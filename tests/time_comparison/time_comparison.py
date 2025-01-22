import torch.distributed as dist
import torch.multiprocessing as mp

from datasets.char_dataset import CharDataset
from src.utils import set_seed


def measure_time(model, trainer_class, dataset, sample_input, sample_target, is_parallel=False):
    trainer_config = trainer_class.get_default_config()

    if is_parallel:
        # Prepare parallelized setup
        model_handler = ModelHandler(model.model_dict, num_subdomains=1, num_replicas_per_subdomain=1)
        trainer_config.model_handler = model_handler
        model = ParallelizedModel(model_handler, sample=sample_input)

    trainer = trainer_class(trainer_config, model, dataset)

    # Measure forward pass
    inputs = sample_input.to(trainer.device)
    targets = sample_target.to(trainer.device)
    start_time = time.time()
    outputs = model(inputs)
    forward_time = time.time() - start_time

    # Reshape outputs and targets for loss computation
    outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * block_size, vocab_size)
    targets = targets.view(-1)  # (batch_size * block_size)

    # Measure backward pass
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs, targets)
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time

    return forward_time, backward_time

def main(rank=None):
    set_seed(3407)  # Ensure reproducibility

    # Initialize the process group
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",  # Or another free port
        rank=rank,
        world_size=1
    )

    text = open('../../input.txt', 'r').read()
    train_dataset = CharDataset(config.data, text)
    sample_input = dataset.get_sample_input(dataset.config)
    sample_target = dataset.get_sample_target(dataset.config)

    # Sequential model setup
    seq_config = SequentialGPT.get_default_config()
    seq_config.vocab_size = dataset.get_vocab_size()
    seq_config.block_size = dataset.get_block_size()
    sequential_model = SequentialGPT(seq_config)

    seq_forward_time, seq_backward_time = measure_time(
        sequential_model, SequentialTrainer, dataset, sample_input, sample_target
    )

    # Parallel model setup
    par_config = ParallelGPT.get_default_config()
    par_config.vocab_size = dataset.get_vocab_size()
    par_config.block_size = dataset.get_block_size()
    parallel_model = ParallelGPT(par_config)

    par_forward_time, par_backward_time = measure_time(
        parallel_model, SequentialTrainer, dataset, sample_input, sample_target, is_parallel=True
    )

    print("Sequential Model:")
    print(f"Forward pass time: {seq_forward_time:.4f} seconds")
    print(f"Backward pass time: {seq_backward_time:.4f} seconds")

    print("\nParallel Model:")
    print(f"Forward pass time: {par_forward_time:.4f} seconds")
    print(f"Backward pass time: {par_backward_time:.4f} seconds")


if __name__ == '__main__':
    world_size = 1
    mp.spawn(
        main,
        args=(),
        nprocs=world_size,
        join=True
    )