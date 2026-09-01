from .toric_code_capacity import ToricCodeCapacity
import torch
import numpy as np
from config.args import Args


class Dataset:
    def __init__(self, args: Args):
        # Noise parameters
        self.error_rate = args.error_rate
        self.error_rates = args.error_rates if args.error_rates else [args.error_rate]
        self.d = args.distance
        self.simulator = ToricCodeCapacity(self.d)
        self.x_logicals = self.simulator.toric_code_x_logicals(self.d)
        self.z_logicals = self.simulator.toric_code_z_logicals(self.d)
        
        # Model parameters
        self.batch_size = args.batch_size
        self.device = args.device
        
    def __getitem__(self):
        if self.error_rates is not None:
            # Randomly select an error rate for this specific sample to mix the batch
            p = np.random.choice(self.error_rates)
        else:
            # Fixed error rate 
            p = self.error_rate
        
        # Sample errors and syndromes
        e_X, e_Z, syn_X, syn_Z = self.simulator.add_depolarizing_code_capacity_noise(p)
        return e_X, e_Z, syn_X, syn_Z

    def generate_batch(self, args):
        """ Generate a batch of syndromes and labels """

        e_X_list, e_Z_list, syn_X_list, syn_Z_list = [], [], [], []
        for _ in range(self.batch_size):
            e_X, e_Z, syn_X, syn_Z = self.__getitem__()
            e_X_list.append(e_X)
            e_Z_list.append(e_Z)
            syn_X_list.append(syn_X)
            syn_Z_list.append(syn_Z)

        # Errors in 1D list of arrays -> Matrix (2L^2, Batch size)
        e_X_mtrx = np.column_stack(e_X_list)
        e_Z_mtrx = np.column_stack(e_Z_list)

        # Pre-Labels (Logical components)
        gamma_x1, gamma_x2, gamma_z1, gamma_z2 = self.create_labels(e_X_mtrx, e_Z_mtrx)

        # Labels (16-class categorical labels)
        labels = (gamma_x1 * 8 + gamma_x2 * 4 + gamma_z1 * 2 + gamma_z2 * 1).astype(np.int64)

        # Reshape syndromes for CNN
        syn_X_np = np.stack(syn_X_list).reshape(args.batch_size, 1, self.d, self.d)
        syn_Z_np = np.stack(syn_Z_list).reshape(args.batch_size, 1, self.d, self.d)
        syndromes = np.concatenate([syn_X_np, syn_Z_np], axis=1)

        # Convert to tensors
        syndromes_tensor = torch.from_numpy(syndromes).to(dtype=torch.float32, device=self.device)
        labels_tensor = torch.from_numpy(labels).to(dtype=torch.long, device=self.device)        

        return syndromes_tensor, labels_tensor
    

    def create_labels(self, e_X_samples, e_Z_samples):
        """ Compute logical components from the sampled errors"""
        gamma_Z = (self.x_logicals @ e_X_samples) % 2
        gamma_X = (self.z_logicals @ e_Z_samples) % 2
        return gamma_X[0], gamma_X[1], gamma_Z[0], gamma_Z[1]


