from gpro.generator.diffusion.diffusion import Diffusion_language
import os

# model training
default_root = "/home/hyu/GPro"
dataset_path = os.path.join(str(default_root), 'data/switch_seq.txt')
checkpoint_path = os.path.join(str(default_root), 'checkpoints/')
model = Diffusion_language(length=60, batch_size=512, transformer_local_size=20)
# model.train(dataset=dataset_path, savepath=checkpoint_path)

# model sampling
sample_model_path = os.path.join(str(default_root), 'checkpoints/diffusion/check/checkpoint.pt')
sample_number = 5000000
model.generate(sample_model_path, sample_number)
