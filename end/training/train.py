import torch
from torch.utils.data import DataLoader
from config.args import Args
from data.data import ToricDataset, ToricCollector
from models.end import End

def train():
    args = Args(
        distance = 5,
        error_rates = [0.001, 0.003, 0.005],
        batch_size = 256,
        n_batches = 256,
        n_epochs=20,
        lr=1e-3
        )
    
    dataset = ToricDataset(args)
    collect = ToricCollector(args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collect,
        num_workers=2,
        prefetch_factor=2 if args.prefetch else None
        )

    model = End().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0.0
        for syndromes, labels in loader:
            preds = model(syndromes)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"epoch {epoch+1}/{args.n_epochs}, loss={total_loss/len(loader):.4f}")

if __name__ == "__main__":
    train()