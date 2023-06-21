import torch
import torch.nn.functional as F

class ModelConfig():
    def __init__(self) -> None:
        pass
    
    def get_depth_runs(self):
        runs = {}
        runs[0] = {
            'hidden_features': [256, 256],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[1] = {
            'hidden_features': [256, 256, 256],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[2] = {
            'hidden_features': [256, 256, 256, 256],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[3] = {
            'hidden_features': [512, 512],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[4] = {
            'hidden_features': [512, 512, 512],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[5] = {
            'hidden_features': [1024, 1024],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[6] = {
            'hidden_features': [1024, 1024, 1024],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        return runs
    
    def get_tiny_runs(self):
        runs = {}
        runs[0] = {
            'hidden_features': [64, 64, 64, 64, 64],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[1] = {
            'hidden_features': [128, 128, 128],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[2] = {
            'hidden_features': [128, 128, 128, 128, 128],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[3] = {
            'hidden_features': [512, 128, 64, 32],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        return runs
    
    def get_paper_comparison_runs(self):
        runs = {}
        runs[0] = {
            'hidden_features': [256, 256, 256],
            'siam_features': [],
            'num_frq': None,
            'first_omega_0': 3000,
            'hidden_omega_0': 30,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[1] = {
            'hidden_features': [256, 256, 256],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 30,
            'hidden_omega_0': 30,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[2] = {
            'hidden_features': [256, 256, 256],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[3] = {
            'hidden_features': [256, 256],
            'siam_features': [128],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        return runs
    
    def get_paper_architecture_runs(self):
        runs = {}
        runs[0] = {
            'hidden_features': [64, 64, 64],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[1] = {
            'hidden_features': [64, 64],
            'siam_features': [32],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[2] = {
            'hidden_features': [128, 128, 128],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[3] = {
            'hidden_features': [128, 128],
            'siam_features': [64],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[4] = {
            'hidden_features': [256, 256, 256],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[5] = {
            'hidden_features': [256, 256],
            'siam_features': [128],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[6] = {
            'hidden_features': [256, 256],
            'siam_features': [64],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[7] = {
            'hidden_features': [512, 512, 256],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[8] = {
            'hidden_features': [512, 512],
            'siam_features': [128],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[9] = {
            'hidden_features': [1024, 1024, 512],
            'siam_features': [256, 256],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        return runs
    
    def get_paper_siam_bonding_runs(self):
        runs = {}
        runs[0] = {
            'hidden_features': [],
            'siam_features': [128, 128, 128],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[1] = {
            'hidden_features': [256],
            'siam_features': [128, 128],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[2] = {
            'hidden_features': [256, 256],
            'siam_features': [128],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        runs[3] = {
            'hidden_features': [256, 256, 256],
            'siam_features': [],
            'num_frq': 16,
            'first_omega_0': 100,
            'hidden_omega_0': 100,
            'optim': torch.optim.Adam,
            'weight_decay': 1e-5,
            'loss_fn': F.mse_loss,
        }
        return runs