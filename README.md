# project_fmri
CS 230 project


vqvae code is based off of https://github.com/MishaLaskin/vqvae/tree/master


To train the VQ-VAE and log results to wandb run train_vqvae.py in the vqvae directory.
A sample run looks like: 

python train_vqvae.py --dataset BLOCK --learning_rate 1e-4 --n_updates 200000 -save

to save results you must use the -save flag and the block.py within vqvae/datasets is
currently configured so that you must se --dataset BLOCK for it to run properly. 

run masked_transformer.py to run masked transformer implementation