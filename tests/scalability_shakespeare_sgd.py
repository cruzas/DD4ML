import warnings
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import utils
from models.nanoGPT import GPTConfig, SequentialGPT
from llms_datasets.tiny_shakespeare import load_shakespeare
import argparse
import time
import pandas as pd
import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader

##########
# TODO: REMOVE AFTER
warnings.filterwarnings("ignore")
#########

bias = True
TEST_ACCURACY = False
hours = 5
total_hours_in_seconds = hours * 60 * 60
save_threshold_in_seconds = total_hours_in_seconds / 2

# Model to read from
model_file = "../saved_networks/tshakespeare_t_0_nsd_2_nrs_1_nst_13_bs_2048_epoch_0.pth"

def main(**kwargs):
    code_start_time = time.time()

    num_epochs = kwargs.get("num_epochs", 40)
    seed = kwargs.get("seed", 2456456)
    batch_size = kwargs.get("batch_size", 64)
    block_size = kwargs.get("block_size", 256)
    n_layer = kwargs.get("n_layer", 1)
    n_head = kwargs.get("n_head", 2)
    n_embd = kwargs.get("n_embd", 384)
    dropout = kwargs.get("dropout", 0.0)
    learning_rate = kwargs.get("learning_rate", 0.1)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_dataset, test_dataset, tokenizer = load_shakespeare(
        train_split=0.8, block_size=block_size)

    config = GPTConfig(
        num_stages=1,  # Not used in sequential mode
        block_size=block_size,
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias
    )

    def criterion(outputs, targets):
        B, T, C = outputs.size()
        loss = F.cross_entropy(outputs.reshape(B * T, C), targets.reshape(-1), ignore_index=-1)
        return loss

    # Create the model
    model = SequentialGPT(config).to(device)

    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    csv_file_name = f"tshakespeare_seq_bs_{batch_size}_sgd.csv"
    iter_csv_file_name = csv_file_name.replace('.csv', '_iter.csv')

    num_iters_per_epoch = len(train_loader)
    print("Number of iterations per epoch: ", num_iters_per_epoch)

    epoch_results = []
    iteration_results = []
    num_iters = 0
    training_start_time = time.time()

    for epoch in range(0, num_epochs + 1):
        model_dict_file_name = csv_file_name.replace(
            '.csv', f'_epoch_{epoch}.pth')

        print(f'____________ EPOCH {epoch} ____________')
        loss_total = 0
        counter = 0
        epoch_start_time = time.time()

        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            counter += 1
            num_iters += 1

            if epoch > 0 and num_iters > 0 and num_iters % 50 == 0:
                running_time = time.time() - training_start_time
                iteration_results.append(
                    {'iteration': num_iters, 'time': running_time, 'loss': loss.item()})
                df_iter_results = pd.DataFrame(iteration_results)
                df_iter_results.to_csv(iter_csv_file_name, index=False)

            code_time_passed = time.time() - code_start_time
            progress = 100 * (i / len(train_loader))

            if (progress > 50 or code_time_passed >= save_threshold_in_seconds) and not os.path.exists(model_dict_file_name):
                torch.save(model.state_dict(), model_dict_file_name)

            if i > 9:  # Early exit for debugging, remove for full training
                break

        avg_loss = loss_total / counter
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch}, avg loss: {avg_loss}, Time: {epoch_time}')

        epoch_results.append(
            {'epoch': epoch, 'time': epoch_time, 'loss': avg_loss})
        df_results = pd.DataFrame(epoch_results)
        df_results.to_csv(csv_file_name, index=False)
        print(f"Results saved to {csv_file_name}")

        torch.save(model.state_dict(), model_dict_file_name)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Script")
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    args = parser.parse_args()

    main(**vars(args))
