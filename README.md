# project_fmri
CS 230 project


vqvae PyTorch architecture and training script is heavily based off https://github.com/MishaLaskin/vqvae/tree/master


VQ-VAE 

To train the VQ-VAE and log results to wandb run train_vqvae.py in the vqvae directory. Adjust hyperparameters as needed
via input arguments To save results you must run with the -save flag.

VQ-VAE Architecture (decoder, encoder, quantizer) is found in vqvae/models

TRANSFORMER 
To train the transformer and log results run masked_transformer.py in the vqvae directory (placed here for file-path convenience). 

Transformer architecture and custom PyTorch dataset is found in vqvae/pipeline_utils.py

VISUALIZER
The code to decode masked and predicted tokens from experiments is found in vqvae/visualize.py. To run visualizations in a jupyter
notebook check out vqvae/visualize.ipynb.

