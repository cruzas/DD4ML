from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import os
import subprocess
import argparse


trainer_config = dict(  max_epochs=300,
                        loss_train_tol=1e-9, 
                        loss_test_tol=1e-9, 
                        acc_train_tol = 99.90, 
                        acc_val_tol = 99.90,     
                        loss_val_tol=1e-5, 
                        loss_diff_tol = 1e-9,     
                        conv_history_csv_name=None, 
                        checkpoint_folder_name=None,
                        checkpoint_file_name=None,
                        resume_from_checkpoint=False,
                        frequency=10)



def setup_distr_env():
    os.environ['MASTER_PORT'] = '29501'
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NNODES']
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = os.environ['SLURM_NODEID']
    node_list = os.environ['SLURM_NODELIST']
    master_node = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1'
    )
    os.environ['MASTER_ADDR'] = master_node



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_params():
    parser = argparse.ArgumentParser(description='SPQN params')
    parser.add_argument('--seed', default=123, type=int, help='seed')

    # discretization related
    parser.add_argument('--num_points_x', default=100, type=int, help='num points in x-dir')    
    parser.add_argument('--num_points_y', default=100, type=int, help='num points in y-dir')    
    parser.add_argument('--num_points_z', default=100, type=int, help='num points in z-dir')    
    parser.add_argument('--num_ref', default=1, type=int, help='num_ref')    


    parser.add_argument('--num_batches', default=1, type=int, help='number of batches to split dataset')    

    
    
    # net config
    parser.add_argument('--width', default=20, type=int, help='width')    
    parser.add_argument('--hiddenlayers_coarse', default=3, type=int, help='hiddenlayers_coarse')    
    parser.add_argument('--T', default=2.0, type=float, help='T')    
    parser.add_argument('--use_adaptive_activation', type=str2bool, nargs='?', default=True, help='use_adaptive_activation')
    parser.add_argument('--num_levels', default=3, type=int, help='width')   
    parser.add_argument('--ada_net', type=str2bool, nargs='?', default=False, help='ada_net') 
    parser.add_argument('--overlap_width', default=0, type=int, help='num of overlap layers')

    parser.add_argument('--model_name', default="none", type=str, help='model name')
    parser.add_argument('--model_name_load', default="none", type=str, help='model name')


    # optimizer related 
    parser.add_argument('--lr_global', default=1e-3, type=float, help='learning_rate global')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--lr_local', default=1e-3, type=float, help='learning_rate local')
    parser.add_argument('--use_SPQN', type=str2bool, nargs='?', default=True, help='Use SPQN preconditioner')
    parser.add_argument('--num_local_steps', default=10, type=int, help='Number of local steps')
    parser.add_argument('--num_coarse_steps', default=5, type=int, help='Number of coarse-level steps')
    parser.add_argument('--use_coarse', type=str2bool, nargs='?', default=True, help='Use coarse-step')
    parser.add_argument('--use_layerwise_lr', type=str2bool, nargs='?', default=True, help='Use layerwise lr')
    parser.add_argument('--use_stochastic', type=str2bool, nargs='?', default=False, help='Use stochastic update')
    parser.add_argument('--freq', default=1, type=int, help='frequency of global step')

    parser.add_argument('--num_subdomains', default=2, type=int, help='Number of num_subdomains')
    parser.add_argument('--history', default=1, type=int, help='History for LBFGS')
    parser.add_argument('--max_epochs', default=1000, type=int, help='Max epochs')
    parser.add_argument('--opt_type', default=1, type=int, help='0-GD(ls), 1-BFGS(ls), 2-Adam, 3-SGD(ls, momentum)')


    args, unknown = parser.parse_known_args()

    return args    





