import os
import sys 

# Absolute path 
current_dir = os.path.dirname(os.path.abspath(__file__))
# Root project folder
parent_dir = os.path.dirname(current_dir)

# Look in the root project folder for modules
sys.path.insert(0, parent_dir)

import argparse
import time
from datetime import datetime
import torch
import numpy as np
from config.args import Args
from data.data import Dataset
from models.end import End
import json

def test():
    # Prepare data to test
    args = Args()
    if args.seed is not None:
        args.seed += 1 # Ensure that the test dataset is different from the training dataset
    cli_args = parse_args()

    # Override default args with command line arguments if provided
    for key, value in vars(cli_args).items():
        if value is not None and hasattr(args, key):
            setattr(args, key, value)
    
    # Load model 
    print(f"Loading model from {args.model_path}")
    model = End(args.init_channels).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    logical_errors_list = [] # Logical error rate for each error rate
    avg_time_per_sample_list = [] # Average time per sample for each error rate
    error_rates = args.error_rates if args.error_rates else [args.error_rate]

    with torch.no_grad():
        for error_rate in error_rates:
            if args.latency:
                # Specifically compute latency
                avg_time_per_sample = latency(error_rate, model, args)
                avg_time_per_sample_list.append(avg_time_per_sample)
                logical_errors_list.append(None)
            else:
                logical_error_rate, avg_time_per_sample = _logical_error_rate(error_rate, args, model)
                logical_errors_list.append(logical_error_rate)
                avg_time_per_sample_list.append(avg_time_per_sample)

    if args.save_results:

        test_results = {
            "metadata": {
                "distance": args.distance,
                "model_path": args.model_path,
                "batch_size": 1 if args.latency else args.batch_size,
                "n_batches": 2000 if args.latency else args.n_batches,
                "total_samples": 2000 if args.latency else (args.batch_size * args.n_batches),
                "timestamp": datetime.now().isoformat(),
                "job": os.environ.get("SLURM_JOB_ID", "local")
            
            },
            "results": {
                "physical_error_rates": error_rates,
                "logical_error_rates": logical_errors_list,
                "avg_time_per_sample": avg_time_per_sample_list
            }
        }

        if args.custom_result_name:
            result_filename = args.custom_result_name
        else:
            run_id = os.environ.get("SLURM_JOB_ID", "") # Slurm job-id

            if not run_id:
                run_id = datetime.now().strftime("%y%m%d_%H%M") # Datetime id

            result_filename = f"test_results_{run_id}.json"
            
        save_dir = args.results_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, result_filename)

        with open(save_path, "w") as f:
            json.dump(test_results, f, indent=4)
            print(f"Results successfully saved to {save_path}")


def _logical_error_rate(error_rate, args, model):
    total_samples = args.batch_size * args.n_batches

    print(f"Testing for error rate: {error_rate}")
    args.error_rate = error_rate
    args.error_rates = None 
    dataset = Dataset(args)
    num_errors = 0
    total_decode_time = 0.0
        
    for batch in range(args.n_batches):
        syndromes, labels = dataset.generate_batch()
        syndromes = syndromes.to(args.device, non_blocking = True)
        labels = labels.to(args.device, non_blocking = True)
        
        # Measure inference time
        if syndromes.is_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # Inference
        logits = model(syndromes)
        pred_classes = torch.argmax(logits, dim=1)

        if syndromes.is_cuda:
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_decode_time += (end_time - start_time)
        num_errors += (labels != pred_classes).sum().item()
    
    logical_error_rate = num_errors / total_samples
    avg_time_per_sample = total_decode_time / total_samples
    print(f"Logical error rate: {logical_error_rate:.6f}, Average time per sample: {avg_time_per_sample:.6f} seconds")
    return logical_error_rate, avg_time_per_sample


def latency(error_rate, model, args):
    args.error_rate = error_rate
    args.error_rates = None 
    args.batch_size = 1
    num_latency_samples = 2000
    dataset = Dataset(args)
    
    print(f"Sampling for error rate: {error_rate}")
    
    # Warm up
    warmup_syndromes, _ = dataset.generate_batch()
    warmup_syndromes = warmup_syndromes.to(args.device)
    for _ in range(50): # Warming up
        _ = model(warmup_syndromes)
    if args.device != 'cpu':
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_latency_samples):
        syndromes, _ = dataset.generate_batch()
        syndromes = syndromes.to(args.device, non_blocking=True)
        
        if syndromes.is_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # Decode
        logits = model(syndromes)
        pred_classes = torch.argmax(logits, dim=1)
        
        if syndromes.is_cuda:
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latencies.append(end_time - start_time)
        
    latencies = np.array(latencies)
    avg_latencies = latencies.mean()
    print(f"Mean Latency: {avg_latencies * 1000:.3f} ms")
    return avg_latencies


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_results", action = "store_true", default =  None, help = "Boolean flag to save the results")
    parser.add_argument("--results_dir", type = str, default = "test_results", help = "Directory to save the results")
    parser.add_argument("--custom_result_name", type = str, default = None, help = "Custom name for the results file")
    parser.add_argument("--model_path", type = str, default = None, help = "Path to load a model")
    parser.add_argument("--latency", action = "store_true", default =  None, help = "Boolean flag to only compute latencies")

    # Hyperparameters
    parser.add_argument("--batch_size", type = int, default = None, help = "Batch size for training")
    parser.add_argument("--n_batches", type = int, default = None, help = "Number of batches per epoch")
    parser.add_argument("--init_channels", type = int, default = None, help = "Number of hidden channels in the model")

    # Toric and noise parameters
    parser.add_argument("--error_rate", type = float, default = None, help = "Error rate for depolarizing-equal_op noise model")
    parser.add_argument("--error_rates", type = float, nargs = "+", default = None, 
                    help = "List of error rates for depolarizing noise (e.g., 0.001 0.003)")
    parser.add_argument("--distance", type = int, default = None, help = "Code distance")
    parser.add_argument("--seed", type = int, dafault = None, help = "Seed to initialize th experiment")
    return parser.parse_args()


if __name__ == "__main__":
    test()