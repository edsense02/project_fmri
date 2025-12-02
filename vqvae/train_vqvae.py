import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
import wandb
from pytorch_msssim import ssim


parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=15000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=64)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset",  type=str, default='BLOCK')

# whether or not to save model
parser.add_argument("-save", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_name = f'newloss_reshiddens{args.n_residual_hiddens}_n_embeddings{args.n_embeddings}_embed_dim{args.embedding_dim}'

wandb.init(
    project="vqvae",
    name=run_name,
    config=args.__dict__
)


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
    'ssim_errors': [], 
    'loss_vals': [],
    'perplexities': [],
    'val_ssim_errors': [],
    'val_recon_errors': [],
    'val_loss_vals': [],
    'val_perplexities': [],
}

def crop_center(img, h=150, w=180):
    B, C, H, W = img.shape
    start_h = (H - h) // 2
    start_w = (W - w) // 2
    return img[:, :, start_h:start_h+h, start_w:start_w+w]


def validate():
    model.eval()
    
    val_recon_errors = []
    val_embedding_losses = []
    val_losses = []
    val_perplexities = []
    val_ssim_errors = []
    
    with torch.no_grad():
        for (x, _) in validation_loader:
            x = x.to(device)
            
            embedding_loss, x_hat, perplexity, tokens = model(x)
            cropped_x = crop_center(x)
            cropped_x_hat = crop_center(x_hat)
            recon_loss = torch.mean((cropped_x_hat - cropped_x)**2) / x_train_var
            ssim_loss = 1 - ssim(cropped_x, cropped_x_hat, data_range=1.0)
            loss = recon_loss + embedding_loss + ssim_loss
            
            val_ssim_errors.append(ssim_loss.cpu().numpy())
            val_recon_errors.append(recon_loss.cpu().numpy())
            val_embedding_losses.append(embedding_loss.cpu().numpy())
            val_losses.append(loss.cpu().numpy())
            val_perplexities.append(perplexity.cpu().numpy())
    
    model.train()
    
    return {
        'val_ssim_error': np.mean(val_ssim_errors),
        'val_recon_error': np.mean(val_recon_errors),
        'val_embedding_loss': np.mean(val_embedding_losses),
        'val_total_loss': np.mean(val_losses),
        'val_perplexity': np.mean(val_perplexities)
    }


def train():
    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity, tokens = model(x)
        cropped_x = crop_center(x)
        cropped_x_hat = crop_center(x_hat)
        recon_loss = torch.mean((cropped_x_hat - cropped_x)**2) / x_train_var
        ssim_loss = 1 - ssim(cropped_x, cropped_x_hat, data_range=1.0)
        loss = recon_loss + embedding_loss + ssim_loss

        loss.backward()
        optimizer.step()

        results["ssim_errors"].append(ssim_loss.cpu().detach().numpy())
        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        # Log to wandb every step (if enabled)
        wandb.log({
                "train ssim error": ssim_loss.item(),
                "train_recon_error": recon_loss.item(),
                "train_embedding_loss": embedding_loss.item(),
                "train_total_loss": loss.item(),
                "train_perplexity": perplexity.item(),
                "update": i
        })

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            avg_ssim = np.mean(results["ssim_errors"][-args.log_interval:])
            avg_recon = np.mean(results["recon_errors"][-args.log_interval:])
            avg_loss = np.mean(results["loss_vals"][-args.log_interval:])
            avg_perp = np.mean(results["perplexities"][-args.log_interval:])

            val_metrics = validate()
            
            results["val_ssim_errors"].append(val_metrics['val_ssim_error'])
            results["val_recon_errors"].append(val_metrics['val_recon_error'])
            results["val_loss_vals"].append(val_metrics['val_total_loss'])
            results["val_perplexities"].append(val_metrics['val_perplexity'])

            print('Update #', i, 
                  'Train Recon Error:', avg_recon,
                  'Train SSIM Error:', avg_ssim, 
                  'Train Loss', avg_loss,
                  'Train Perplexity:', avg_perp)
            print('          ',
                  'Val SSIM Error:', val_metrics['val_ssim_error'],
                  'Val Recon Error:', val_metrics['val_recon_error'],
                  'Val Loss', val_metrics['val_total_loss'],
                  'Val Perplexity:', val_metrics['val_perplexity'])

            # Log averaged training metrics and validation metrics to wandb
            wandb.log({
                    f"avg_train_ssim_error_{args.log_interval}": avg_ssim,
                    f"avg_train_recon_error_{args.log_interval}": avg_recon,
                    f"avg_train_loss_{args.log_interval}": avg_loss,
                    f"avg_train_perplexity_{args.log_interval}": avg_perp,
                    "val_ssim_error": val_metrics['val_ssim_error'],
                    "val_recon_error": val_metrics['val_recon_error'],
                    "val_embedding_loss": val_metrics['val_embedding_loss'],
                    "val_total_loss": val_metrics['val_total_loss'],
                    "val_perplexity": val_metrics['val_perplexity'],
            })
    
    if args.save:
        hyperparameters = args.__dict__
        utils.save_model_and_results(
            model, results, hyperparameters, run_name)

    wandb.finish()


if __name__ == "__main__":
    train()