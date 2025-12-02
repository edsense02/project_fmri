'''
An MRI sample is a 3D sequence of 2D slices. This script tokenizes each 2D slice into 
4096 tokens and trains a transformer encoder to accurately reconstruct masked out tokens 
'''

import torch
import torch.nn as nn
from models.vqvae import VQVAE
from torch.utils.data import DataLoader
from pipeline_utils import MRITokenDataset, MRITransformer
from accelerate import Accelerator
import argparse
import numpy as np
import wandb
import os

IGNORE_VAL = -1 

def tokenize(checkpoint_name, h_dim=128, res_h_dim=32, n_res_layers=2, n_embeddings=64, embedding_dim=16, beta=0.25): 
    '''
    Returns tokens for training, validation, and test. Each axial slice is 4096 tokens and each MRI sample has 193 slices
    '''
    model_path = f'/home/mingjie/mri230/vqvae_checkpoints/{checkpoint_name}'
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    slices_per_sample = 193 
    num_train_samples = 72
    num_val_samples = 15
    num_test_samples = 16 
    
    model = VQVAE(
        h_dim=h_dim, res_h_dim=res_h_dim, n_res_layers=n_res_layers,
        n_embeddings=n_embeddings, embedding_dim=embedding_dim, beta=beta
    )
    
    model.load_state_dict(checkpoint["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Extract training tokens 
    train_data = np.load("/home/mingjie/mri230/train_data/train_data.npy")    
    train_tokens_lst = []
    for i in range(num_train_samples): 
        train_sample = torch.from_numpy(train_data[i * slices_per_sample : (i + 1) * slices_per_sample]).float().to(device)   
        with torch.no_grad():
            _, _, _, train_tokens = model(train_sample)
        train_tokens_lst.append(train_tokens.squeeze().cpu().numpy())
    train_token_seq = np.concatenate(train_tokens_lst)
    
    # Extract validation and test tokens 
    val_data = np.load("/home/mingjie/mri230/val_data/val_data.npy")    
    val_tokens_lst = []
    test_tokens_lst = []
    for i in range(num_val_samples + num_test_samples): 
        if i < num_val_samples: 
            val_sample = torch.from_numpy(val_data[i * slices_per_sample : (i + 1) * slices_per_sample]).float().to(device)   
            with torch.no_grad():
                _, _, _, val_tokens = model(val_sample)
            val_tokens_lst.append(val_tokens.squeeze().cpu().numpy())
        else: 
            test_sample = torch.from_numpy(val_data[i * slices_per_sample : (i + 1) * slices_per_sample]).float().to(device)   
            with torch.no_grad():
                _, _, _, test_tokens = model(test_sample)
            test_tokens_lst.append(test_tokens.squeeze().cpu().numpy())
    val_token_seq = np.concatenate(val_tokens_lst)
    test_token_seq = np.concatenate(test_tokens_lst)
    
    return train_token_seq, val_token_seq, test_token_seq


def train(args): 
    
    model_name = f'newloss_epochs{args.epochs}_maskprob{int(args.mask_prob * 10)}_embdim{args.embedding_dim}'
    wandb.init(project="transformerMRI", name=model_name)
    
    context_slices = args.context_slices
    mask_prob = args.mask_prob
    
    # set up batch accumulator 
    batch_size = args.batch_size
    accelerator = Accelerator(mixed_precision="fp16")
    
    # obtain tokens 
    checkpoint_name = args.checkpoint_name
    train_token_seq, val_token_seq, test_token_seq = tokenize(checkpoint_name)
    
    # set up datasets 
    train_dataset = MRITokenDataset(tokens=train_token_seq, context_slices=context_slices, mask_prob=mask_prob)
    val_dataset = MRITokenDataset(tokens=val_token_seq, context_slices=context_slices, mask_prob=mask_prob)
    test_dataset = MRITokenDataset(tokens=test_token_seq, context_slices=context_slices, mask_prob=mask_prob)
    
    # set up dataloaders 
    num_workers = 4 
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # hyperparameters 
    learning_rate = args.lr
    sequence_len = 4096 * context_slices
    embedding_dim = args.embedding_dim
    vocab_size = args.vocab_size
    
    model = MRITransformer(sequence_len=sequence_len, vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_VAL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader)
    
    # training loop
    epochs = args.epochs
    for epoch in range(epochs):
        
        # train
        model.train()
        train_loss = 0
        train_correct_predictions = 0
        train_total_predictions = 0
        for x, labels in train_dataloader:
            x = x.to(device)    # (batch_size, sequence_length)
            labels = labels.to(device)  # (batch_size, sequence_length)
            logits = model(x)   # (batch_size, sequence_length, vocab_size)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            
            # calculate accuracy
            masked_token_positions = (labels != IGNORE_VAL)
            number_of_predictions = masked_token_positions.sum().item()
            if number_of_predictions > 0:
                predictions = logits.argmax(dim=-1)  # get token with highest predicted probability 
                train_correct_predictions += (predictions[masked_token_positions] == labels[masked_token_positions]).sum().item()
                train_total_predictions += number_of_predictions
            
        train_loss /= len(train_dataloader)
        train_accuracy = train_correct_predictions / train_total_predictions if train_total_predictions > 0 else 0
        wandb.log({"epoch": epoch, "train loss": train_loss, "train accuracy": train_accuracy})
        
        # validate 
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_correct_predictions = 0
            val_total_predictions = 0
            for x, labels in val_dataloader:
                x = x.to(device)    # (batch_size, sequence_length)
                labels = labels.to(device)  # (batch_size, sequence_length)
                logits = model(x)   # (batch_size, sequence_length, vocab_size)
                loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
                val_loss += loss.item()
                
                # calculate accuracy
                masked_token_positions = (labels != IGNORE_VAL)
                number_of_predictions = masked_token_positions.sum().item()
                if number_of_predictions > 0:
                    predictions = logits.argmax(dim=-1)  # get token with highest predicted probability 
                    val_correct_predictions += (predictions[masked_token_positions] == labels[masked_token_positions]).sum().item()
                    val_total_predictions += number_of_predictions
                    
            val_loss /= len(val_dataloader)
            val_accuracy = val_correct_predictions / val_total_predictions if val_total_predictions > 0 else 0
            wandb.log({"epoch": epoch, "validation loss": val_loss, "validation accuracy": val_accuracy})
        
        # print outputs
        print(f"Epoch {epoch}: train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, train acc: {train_accuracy:.4f}, val acc: {val_accuracy:.4f}")
    
    # save model
    if accelerator.is_main_process: 
        save_dir = '/home/mingjie/mri230/transformer_checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(save_dir, model_name + '.pth')
        torch.save({
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'args': vars(args)}, save_path)
        
    # test
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_correct_predictions = 0
        test_total_predictions = 0
        for x, labels in test_dataloader:
            x = x.to(device)    # (batch_size, sequence_length)
            labels = labels.to(device)  # (batch_size, sequence_length)
            logits = model(x)   # (batch_size, sequence_length, vocab_size)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            test_loss += loss.item()
            
            # calculate accuracy
            masked_token_positions = (labels != IGNORE_VAL)
            number_of_predictions = masked_token_positions.sum().item()
            if number_of_predictions > 0:
                predictions = logits.argmax(dim=-1)  # get token with highest predicted probability 
                test_correct_predictions += (predictions[masked_token_positions] == labels[masked_token_positions]).sum().item()
                test_total_predictions += number_of_predictions
            
        test_loss /= len(test_dataloader)
        test_accuracy = test_correct_predictions / test_total_predictions if test_total_predictions > 0 else 0
        wandb.log({"test loss": test_loss, "test accuracy": test_accuracy})
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--context_slices', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mask_prob', type=float, default=0.25)
    parser.add_argument('--vocab_size', type=int, default=65)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--checkpoint_name', type=str, default='newloss_reshiddens32_n_embeddings64_embed_dim16.pth')
    args = parser.parse_args()
    train(args)