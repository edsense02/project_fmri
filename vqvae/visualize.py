import numpy as np
import torch
import json
from models.vqvae import VQVAE
from masked_transformer import *
from pipeline_utils import *
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

VISUAL_MASK_ID = 63
MASK_ID = 64

'''
Outputs reconstructed image from input vqvae and latent space embeddings
'''
def decode_from_embeddings(vqvae, embeddings):
    B, e_dim = 1, embeddings.shape[-1]
    H = W = int(np.sqrt(embeddings.shape[0])) 
    z_q = embeddings.view(H, W, e_dim)      # (64, 64, 32)
    z_q = z_q.permute(2, 0, 1).contiguous() # (32, 64, 64)
    z_q = z_q.unsqueeze(0)                  # (1, 32, 64, 64)
    with torch.no_grad():
        x_hat = vqvae.decoder(z_q)          # (1, 1, 256, 256)
    return x_hat

'''
Crops data to center around relevant brain image pixels 
'''
def crop_center(img, h=150, w=180):
    H, W = img.shape
    start_h = (H - h) // 2
    start_w = (W - w) // 2
    return img[start_h:start_h+h, start_w:start_w+w]

'''
Class to visualize MRI slices
'''
class VisualizeSlicesMRI:
    def __init__(self, vqvae_checkpoint, transformer_checkpoint, mask_prob, subject):
        res_h_dim = 32
        embedding_dim = 16
        n_res_layers = 2
        n_embeddings = 64
        
        # load vqvae checkpoint and obtain validation tokens
        vqvae_checkpoint_path = f'/home/mingjie/mri230/vqvae_checkpoints/{vqvae_checkpoint}'
        self.train_token_seq, self.val_token_seq, self.test_token_seq = tokenize(vqvae_checkpoint, res_h_dim=res_h_dim, embedding_dim=embedding_dim)
        vqvae_checkpoint_model = torch.load(vqvae_checkpoint_path, map_location="cpu", weights_only=False)
        self.vqvae = VQVAE(h_dim=128, res_h_dim=res_h_dim, n_res_layers=n_res_layers, n_embeddings=n_embeddings, embedding_dim=embedding_dim, beta=0.25)
        self.vqvae.load_state_dict(vqvae_checkpoint_model["model"])
        self.embedding_weight = self.vqvae.vector_quantization.embedding.weight 
        
        # select random subject and obtain quartile slices for visualization
        self.tokens_per_slice = 4096 
        self.tokens_per_subject = self.tokens_per_slice * 193
        self.first_q_idx = 193 // 4
        self.second_q_idx = 193 // 2
        self.third_q_idx = self.first_q_idx * 3
        self.subject = subject - 1
        self.subject_tokens = self.val_token_seq[self.tokens_per_subject * self.subject : self.tokens_per_subject * (self.subject + 1)]
        
        # get mask and masked tokens for visualization 
        self.mask_prob = mask_prob
        self.mask = np.random.rand(self.subject_tokens.shape[0]) < mask_prob
        self.subject_tokens_masked = self.subject_tokens.copy()
        self.subject_tokens_masked[self.mask] = VISUAL_MASK_ID
        
        # load transformer checkpoint 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer_checkpoint_path = f'/home/mingjie/mri230/transformer_checkpoints/{transformer_checkpoint}'
        transformer_checkpoint_weights = torch.load(transformer_checkpoint_path)
        self.transformer = MRITransformer(sequence_len=4096 * 3, embedding_dim=256).to(self.device)
        self.transformer.load_state_dict(transformer_checkpoint_weights['model_state_dict'])
        self.transformer.eval()

        
    def plot_nominal(self, plot=True): 
        first_q_slice_tokens = self.subject_tokens[self.tokens_per_slice * self.first_q_idx : self.tokens_per_slice * (self.first_q_idx + 1)]
        second_q_slice_tokens = self.subject_tokens[self.tokens_per_slice * self.second_q_idx : self.tokens_per_slice * (self.second_q_idx + 1)]
        third_q_slice_tokens = self.subject_tokens[self.tokens_per_slice * self.third_q_idx : self.tokens_per_slice * (self.third_q_idx + 1)]
        
        first_q_gt_img = crop_center(decode_from_embeddings(self.vqvae, self.embedding_weight[first_q_slice_tokens]).squeeze().numpy())
        second_q_gt_img = crop_center(decode_from_embeddings(self.vqvae, self.embedding_weight[second_q_slice_tokens]).squeeze().numpy())
        third_q_gt_img = crop_center(decode_from_embeddings(self.vqvae, self.embedding_weight[third_q_slice_tokens]).squeeze().numpy())
        
        if plot: 
            _, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(first_q_gt_img, cmap="gray")
            axes[0].set_title("25% Axial Depth")
            axes[0].axis("off")
            
            axes[1].imshow(second_q_gt_img, cmap="gray")
            axes[1].set_title("50% Axial Depth")
            axes[1].axis("off")
            
            axes[2].imshow(third_q_gt_img, cmap="gray")
            axes[2].set_title("75% Axial Depth")
            axes[2].axis("off")
        
            plt.tight_layout()
            plt.show()
        else: 
            return first_q_gt_img, second_q_gt_img, third_q_gt_img
        
        
    def plot_masked(self, plot=True):
        first_q_slice_tokens_masked = self.subject_tokens_masked[self.tokens_per_slice * self.first_q_idx : self.tokens_per_slice * (self.first_q_idx + 1)]
        second_q_slice_tokens_masked = self.subject_tokens_masked[self.tokens_per_slice * self.second_q_idx : self.tokens_per_slice * (self.second_q_idx + 1)]
        third_q_slice_tokens_masked = self.subject_tokens_masked[self.tokens_per_slice * self.third_q_idx : self.tokens_per_slice * (self.third_q_idx + 1)]
        
        first_q_gt_img_masked = crop_center(decode_from_embeddings(self.vqvae, self.embedding_weight[first_q_slice_tokens_masked]).squeeze().numpy())
        second_q_gt_img_masked = crop_center(decode_from_embeddings(self.vqvae, self.embedding_weight[second_q_slice_tokens_masked]).squeeze().numpy())
        third_q_gt_img_masked = crop_center(decode_from_embeddings(self.vqvae, self.embedding_weight[third_q_slice_tokens_masked]).squeeze().numpy())
        
        if plot: 
            _, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(first_q_gt_img_masked, cmap="gray")
            axes[0].set_title("25% Axial Depth Masked")
            axes[0].axis("off")
            
            axes[1].imshow(second_q_gt_img_masked, cmap="gray")
            axes[1].set_title("50% Axial Depth Masked")
            axes[1].axis("off")
            
            axes[2].imshow(third_q_gt_img_masked, cmap="gray")
            axes[2].set_title("75% Axial Depth Masked")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.show()
        else: 
            return first_q_gt_img_masked, second_q_gt_img_masked, third_q_gt_img_masked
    
    
    def plot_masked_predict(self, plot=True): 
        subject_tokens_masked_input = self.subject_tokens_masked.copy()
        subject_tokens_masked_input[self.mask] = MASK_ID
        
        first_q_slice_tokens_masked_prev = subject_tokens_masked_input[self.tokens_per_slice * (self.first_q_idx - 1) : self.tokens_per_slice * self.first_q_idx]
        first_q_slice_tokens_masked_input = subject_tokens_masked_input[self.tokens_per_slice * self.first_q_idx : self.tokens_per_slice * (self.first_q_idx + 1)]
        first_q_slice_tokens_masked_next = subject_tokens_masked_input[self.tokens_per_slice * (self.first_q_idx + 1) : self.tokens_per_slice * (self.first_q_idx + 2)]
        first_q_mask = first_q_slice_tokens_masked_input != MASK_ID
        
        second_q_slice_tokens_masked_prev = subject_tokens_masked_input[self.tokens_per_slice * (self.second_q_idx - 1) : self.tokens_per_slice * self.second_q_idx]
        second_q_slice_tokens_masked_input = subject_tokens_masked_input[self.tokens_per_slice * self.second_q_idx : self.tokens_per_slice * (self.second_q_idx + 1)]
        second_q_slice_tokens_masked_next = subject_tokens_masked_input[self.tokens_per_slice * (self.second_q_idx + 1) : self.tokens_per_slice * (self.second_q_idx + 2)]
        second_q_mask = second_q_slice_tokens_masked_input != MASK_ID
        
        third_q_slice_tokens_masked_prev = subject_tokens_masked_input[self.tokens_per_slice * (self.third_q_idx - 1) : self.tokens_per_slice * self.third_q_idx]
        third_q_slice_tokens_masked_input = subject_tokens_masked_input[self.tokens_per_slice * self.third_q_idx : self.tokens_per_slice * (self.third_q_idx + 1)]
        third_q_slice_tokens_masked_next = subject_tokens_masked_input[self.tokens_per_slice * (self.third_q_idx + 1) : self.tokens_per_slice * (self.third_q_idx + 2)]
        third_q_mask = third_q_slice_tokens_masked_input != MASK_ID
        
        first_q_tokens_masked = torch.tensor(np.concatenate([first_q_slice_tokens_masked_prev, first_q_slice_tokens_masked_input, first_q_slice_tokens_masked_next])).to(self.device)
        second_q_tokens_masked = torch.tensor(np.concatenate([second_q_slice_tokens_masked_prev, second_q_slice_tokens_masked_input, second_q_slice_tokens_masked_next])).to(self.device)
        third_q_tokens_masked = torch.tensor(np.concatenate([third_q_slice_tokens_masked_prev, third_q_slice_tokens_masked_input, third_q_slice_tokens_masked_next])).to(self.device)
        
        with torch.no_grad():
            first_q_logits = self.transformer(first_q_tokens_masked.unsqueeze(0))
            first_q_predictions = first_q_logits.argmax(dim=-1).squeeze().cpu()
            
            second_q_logits = self.transformer(second_q_tokens_masked.unsqueeze(0))
            second_q_predictions = second_q_logits.argmax(dim=-1).squeeze().cpu()
            
            third_q_logits = self.transformer(third_q_tokens_masked.unsqueeze(0))
            third_q_predictions = third_q_logits.argmax(dim=-1).squeeze().cpu()
        
        first_q_slice = first_q_predictions[self.tokens_per_slice : 2 * self.tokens_per_slice].numpy()
        second_q_slice = second_q_predictions[self.tokens_per_slice : 2 * self.tokens_per_slice].numpy()
        third_q_slice = third_q_predictions[self.tokens_per_slice : 2 * self.tokens_per_slice].numpy()
        
        first_q_slice[first_q_mask] = first_q_slice_tokens_masked_input[first_q_mask]
        second_q_slice[second_q_mask] = second_q_slice_tokens_masked_input[second_q_mask]
        third_q_slice[third_q_mask] = third_q_slice_tokens_masked_input[third_q_mask]
        
        first_q_gt_img_pred = crop_center(decode_from_embeddings(self.vqvae, self.embedding_weight[first_q_slice]).squeeze().numpy())
        second_q_gt_img_pred = crop_center(decode_from_embeddings(self.vqvae, self.embedding_weight[second_q_slice]).squeeze().numpy())
        third_q_gt_img_pred = crop_center(decode_from_embeddings(self.vqvae, self.embedding_weight[third_q_slice]).squeeze().numpy())
        
        if plot:
            _, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(first_q_gt_img_pred, cmap="gray")
            axes[0].set_title("25% Axial Depth Predicted")
            axes[0].axis("off")
            
            axes[1].imshow(second_q_gt_img_pred, cmap="gray")
            axes[1].set_title("50% Axial Depth Predicted")
            axes[1].axis("off")
            
            axes[2].imshow(third_q_gt_img_pred, cmap="gray")
            axes[2].set_title("75% Axial Depth Predicted")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.show()
        else:
            return first_q_gt_img_pred, second_q_gt_img_pred, third_q_gt_img_pred
    
    
def output_metrics(mask_prob):
    v_check = 'newloss_reshiddens32_n_embeddings64_embed_dim16.pth'
    t_check = f'epochs100_maskprob{int(mask_prob * 100)}_embdim256.pth' 
    
    mse_dict = defaultdict(list)
    ssim_dict = defaultdict(list)
    
    for subject in range(1, 16):
        vis = VisualizeSlicesMRI(v_check, t_check, mask_prob, subject)
        first_q_nominal, second_q_nominal, third_q_nominal = vis.plot_nominal(plot=False)
        first_q_masked, second_q_masked, third_q_masked = vis.plot_masked(plot=False)
        first_q_recon, second_q_recon, third_q_recon = vis.plot_masked_predict(plot=False)
        
        mse_dict["masked_first"].append(np.mean((first_q_nominal - first_q_masked) ** 2))
        mse_dict["masked_second"].append(np.mean((second_q_nominal - second_q_masked) ** 2))
        mse_dict["masked_third"].append(np.mean((third_q_nominal - third_q_masked) ** 2))
        
        mse_dict["recon_first"].append(np.mean((first_q_nominal - first_q_recon) ** 2))
        mse_dict["recon_second"].append(np.mean((second_q_nominal - second_q_recon) ** 2))
        mse_dict["recon_third"].append(np.mean((third_q_nominal - third_q_recon) ** 2))
        
        ssim_dict["masked_first"].append(ssim(first_q_nominal, first_q_masked, data_range=first_q_nominal.max() - first_q_masked.min()))
        ssim_dict["masked_second"].append(ssim(second_q_nominal, second_q_masked, data_range=second_q_nominal.max() - second_q_masked.min()))
        ssim_dict["masked_third"].append(ssim(third_q_nominal, third_q_masked, data_range=third_q_nominal.max() - third_q_masked.min()))
        
        ssim_dict["recon_first"].append(ssim(first_q_nominal, first_q_recon, data_range=first_q_nominal.max() - first_q_recon.min()))
        ssim_dict["recon_second"].append(ssim(second_q_nominal, second_q_recon, data_range=second_q_nominal.max() - second_q_recon.min()))
        ssim_dict["recon_third"].append(ssim(third_q_nominal, third_q_recon, data_range=third_q_nominal.max() - third_q_recon.min()))
    
    avg_mse = {key: float(np.mean(vals)) for key, vals in mse_dict.items()}
    avg_ssim = {key: float(np.mean(vals)) for key, vals in ssim_dict.items()}

    save_path = f"metrics_maskprob{int(mask_prob*100)}.json"
    with open(save_path, "w") as f:
        json.dump({"mse": avg_mse, "ssim": avg_ssim}, f, indent=4)
        

if __name__ == "__main__":
    output_metrics(0.15)
    output_metrics(0.25)
    output_metrics(0.5)
    output_metrics(0.75)