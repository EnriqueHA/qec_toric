import os
import sys 

# Absolute path 
current_dir = os.path.dirname(os.path.abspath(__file__))
# Root project folder
parent_dir = os.path.dirname(current_dir)

# Look in the root project folder for modules
sys.path.insert(0, parent_dir)

import argparse
from datetime import datetime
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from lion_pytorch import Lion
from muon import Muon
import torch.distributed as dist
from config.args import Args
from data.data import Dataset
from models.end import End

def train():
    args = Args()
    cli_args = parse_args()

    # Override default args with command line arguments if provided
    for key, value in vars(cli_args).items():
        if value is not None and hasattr(args, key):
            setattr(args, key, value)

    # Define single GPU for Muon
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # Arbitrary open port
        dist.init_process_group(
            backend = 'nccl' if torch.cuda.is_available() else 'gloo', 
            rank = 0, 
            world_size = 1
        )

    dataset = Dataset(args)

    model = End(args.init_channels).to(args.device)
    # Optimizers
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    lion_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    optimizer_muon = Muon(muon_params, lr = args.lr_muon, weight_decay = args.weight_decay)
    optimizer_lion = Lion(lion_params, lr = args.lr_lion, weight_decay = args.weight_decay)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Scheduler
    scheduler_muon = scheduler(optimizer_muon, args, args.lr_muon_max)
    scheduler_lion = scheduler(optimizer_lion, args, args.lr_lion_max)

    # Init train
    total_steps = args.n_epochs * args.n_batches
    total_samples_epoch = args.batch_size * args.n_batches

    print(f"Code distance: {args.distance}")
    print(f"Error rate: {args.error_rate if args.error_rate is not None else args.error_rates}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params   : {n_params:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches per epoch: {args.n_batches}")
    print(f"Dataset size: {total_samples_epoch*args.n_epochs}")
    print(f"Learning rates: Muon = {args.lr_muon}, Lion = {args.lr_lion}")
    print(f"Weight decay = {args.weight_decay}")

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0.0
        preds_correct = 0

        for batch in range(args.n_batches):
            syndromes, labels = dataset.generate_batch()
            syndromes = syndromes.to(args.device, non_blocking = True)
            labels = labels.to(args.device, non_blocking = True)

            optimizer_muon.zero_grad()
            optimizer_lion.zero_grad()
            logits = model(syndromes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_muon.step()
            optimizer_lion.step()
            total_loss += loss.item()

            pred_classes = torch.argmax(logits, dim=1)
            preds_correct += (pred_classes == labels).sum().item()

            scheduler_muon.step()
            scheduler_lion.step()
        
        avg_loss = total_loss / args.n_batches
        epoch_accuracy = preds_correct / total_samples_epoch
            
        print(f"epoch {epoch+1}/{args.n_epochs}, loss = {avg_loss:.5f}, accuracy = {epoch_accuracy:.5f}")

        # Save checkpoint
        if args.save_model:

            checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict_muon": optimizer_muon.state_dict(),
                    "optimizer_state_dict_lion": optimizer_lion.state_dict(),
                    "learning_rate_muon": optimizer_muon.param_groups[0]['lr'],
                    "learning_rate_lion": optimizer_lion.param_groups[0]['lr'],
                    "loss": avg_loss,
                    "accuracy": epoch_accuracy,
                    "job": os.environ.get("SLURM_JOB_ID", "local") 
                }

            run_id = os.environ.get("SLURM_JOB_ID", "") # Slurm job-id

            if not run_id:
                    run_id = datetime.now().strftime("%y%m%d_%H%M") # Datetime id

            # Save best checkpoint
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy

                checkpoint_filename = f"best_checkpoint_toric_{run_id}.pt" 
                save_checkpoints(checkpoint, args.checkpoint_dir, checkpoint_filename)

            latest_filename = f"latest_checkpoint_toric_{run_id}.pt"
            save_checkpoints(checkpoint, args.checkpoint_dir, latest_filename)

def scheduler(optimizer, args, lr):
    """ Defines the learning rate schedule"""
    # To make it smooth, we use steps to change the schedule instead of epochs
    epochs_warmup = args.epochs_warmup if hasattr(args, "epochs_warmup") else args.n_epochs // 10
    steps_warmup = epochs_warmup * args.n_batches
    total_steps = args.n_epochs * args.n_batches
    steps_cosine = total_steps - steps_warmup

    scheduler_warmup = LinearLR(
        optimizer, 
        start_factor = 0.2*lr, 
        end_factor = 1.0,
        total_iters = steps_warmup
    )
    scheduler_cosine = CosineAnnealingLR(
        optimizer, 
        T_max = steps_cosine, 
        eta_min = 0.1*lr
    )

    scheduler = SequentialLR(
        optimizer, 
        schedulers = [scheduler_warmup, scheduler_cosine], 
        milestones = [steps_warmup]
    )

    return scheduler


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    # Management
    parser.add_argument("--save_model", action = "store_true", default =  None, help = "Boolean flag to save the model")
    parser.add_argument("--checkpoint_dir", type = str, default = None, help = "Directory to save the model checkpoints")

    # Hyperparameters
    parser.add_argument("--batch_size", type = int, default = None, help = "Batch size for training")
    parser.add_argument("--n_batches", type = int, default = None, help = "Number of batches per epoch")
    parser.add_argument("--n_epochs", type = int, default = None, help = "Number of epochs for training")
    parser.add_argument("--epochs_warmup", type = int, default = None, help = "Number of initial epochs to warmup")
    parser.add_argument("--lr_muon", type = float, default = None, help = "Maximum learning rate for the Muon optimizer")
    parser.add_argument("--lr_lion", type = float, default = None, help = "Maximum learning rate for the Lion optimizer")
    parser.add_argument("--weight_decay", type = float, default = None, help = "Weight decay")
    parser.add_argument("--init_channels", type = int, default = None, help = "Number of hidden channels in the model")
    
    # Toric and noise parameters
    parser.add_argument("--error_rate", type = float, default = None, help = "Error rate for depolarizing-equal_op noise model")
    parser.add_argument("--error_rates", type = float, nargs = "+", default = None, 
                    help = "List of error rates for depolarizing noise (e.g., 0.001 0.003)")
    parser.add_argument("--distance", type = int, default = None, help = "Code distance")
    parser.add_argument("--seed", type = int, dafault = None, help = "Seed to initialize th experiment")
    return parser.parse_args()


def save_checkpoints(checkpoint, directory, filename):
    os.makedirs(directory, exist_ok = True)
    filepath = os.path.join(directory, filename)
    torch.save(checkpoint, filepath)

if __name__ == "__main__":
    train()