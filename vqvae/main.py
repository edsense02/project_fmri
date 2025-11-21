import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
import wandb

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset",  type=str, default='CIFAR10')

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project="vqvae",
    name=f'vqvae_lr{args.learning_rate}_beta{args.beta}_n_embeddings{args.n_embeddings}',
    config=args.__dict__
)

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)

"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}


def train():

    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity, tokens = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        # Log to wandb every step (if enabled)
        wandb.log({
                "recon_error": recon_loss.item(),
                "embedding_loss": embedding_loss.item(),
                "total_loss": loss.item(),
                "perplexity": perplexity.item(),
                "update": i
        })

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            avg_recon = np.mean(results["recon_errors"][-args.log_interval:])
            avg_loss = np.mean(results["loss_vals"][-args.log_interval:])
            avg_perp = np.mean(results["perplexities"][-args.log_interval:])

            print('Update #', i, 'Recon Error:', avg_recon,
                  'Loss', avg_loss,
                  'Perplexity:', avg_perp)

            # Log averaged metrics to wandb (if enabled)
            wandb.log({
                    f"avg_recon_error_{args.log_interval}": avg_recon,
                    f"avg_loss_{args.log_interval}": avg_loss,
                    f"avg_perplexity_{args.log_interval}": avg_perp,
            })
    
    if args.save:
        hyperparameters = args.__dict__
        utils.save_model_and_results(
            model, results, hyperparameters, args.filename)

    wandb.finish()


if __name__ == "__main__":
    train()