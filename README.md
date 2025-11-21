# project_fmri
CS 230 project


vqvae code is based off of https://github.com/MishaLaskin/vqvae/tree/master



To train the VQ-VAE and log results to wandb run main.py in the vqvae directory.
A sample run looks like: 

python main.py --dataset BLOCK --learning_rate 1e-4 --n_updates 200000 -save

to save results you must use the -save flag and the block.py within vqvae/datasets is
currently configured so that you must se --dataset BLOCK for it to run properly. 


I recommend taking a close look at examine.ipynb to see how I organized the data and then
vqvae/utils.py to see where you need to modify paths in the load_block, load_latent_block,
and save_model_and_results functions.  

vqvae/extract_tokens.py takes the latest trained vqvae and run encoding to get tokens. 