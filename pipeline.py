import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

Width = 512
Height = 512
Latent_Width = Width // 8
Latent_Height = Height // 8

# more noise more creative less noise less creative

# cfg_scale = 7.5 (difference between conditioned and unconditioned)
# uncon_prompt = without any conditontion
# each time step indicate the noise level so 

def generate(prompt : str , uncon_prompt: str , inp_image = None , 
             strength = 0.8 , do_cfg= True , sampler_name = "ddpm" , 
             n_inferenc_steps = 50 , model={}, seed = None,
             device = None , idle_device = None , tokenizer = None,
             cfg_scale = 7.5
             ):
    with torch.no_grad():

        if not (0 < strength <= 1):
                raise ValueError("strength must be between 0 an d1")
        if idle_device:
             to_idle : lambda x : x.to(idle_device)
        else:
             to_idle: lambda x : x
        
        generate = torch.Generator(device=device)
        if seed is None:
             generate.seed()
        else: 
             generate.manual_seed(seed)

        clip = model["clip"]
        clip.to(device)

        if do_cfg:
             #convert the prompt into tokens using otkinzer
             cond_tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length" , max_length = 77).input_ids

             cond_tokens = torch.tensor(cond_tokens, dtype=torch.long , device = device)

             cond_context = clip(cond_tokens)

             uncon_tokens = tokenizer.batch_encode_plus([uncon_prompt], padding ="max_length", max_length = 77).input_ids
             uncon_tokens = torch.tensor(uncon_tokens, dtype=torch.long , device=device)

             uncon_context = clip(uncon_tokens)

             context = torch.cat([cond_context, uncon_context])
        
        else:
             cond_tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length" , max_length = 77).input_ids
             tokens = torch.tensor(tokens , dtype=torch.long , device = device)

             context= clip(tokens)

        to_idle(clip) #for moving the model to the cpu

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generate)
            sampler.set_inferrence_steps(n_inferenc_steps)
        else:
            raise ValueError
        latents_shape = (1,4,Latent_Height, Latent_Width)

        if inp_image:
             encoder = model["encoder"]
             encoder.to(device)

             input_image_tensor = input_image_tensor.resize(Height, Width)
             input_image_tensor = np.array(input_image_tensor)

             input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
             
             input_image_tensor = input_image_tensor.unsqueeze(0)
             input_image_tensor =  input_image_tensor.permute(0,3,1,2)

             encoder_noise = torch.randn(latents_shape, generator=generate , device=device)

             latents = encoder(input_image_tensor, encoder_noise)

             sampler.set_strength(strength= strength)
             latents = sampler.add_noise(latents, sampler.timesteps[0])

             to_idle(encoder)

        else:
             latents = torch.randn(latents_shape , generator=generate, device=device)

        diffusion = model["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
         
        for i , timestep in enumerate(timesteps):
             time_embedding = get_time_embedding(timestep).to(device)

             model_input = latents

             if do_cfg:
                  model_input = model_input.repeat(2,1,1,1)
                

             model_output = diffusion(model_input, context, time_embedding)


            #  why do we predict noise instead of the denoised image itself

             if do_cfg:
                  output_cond , output_uncond = model_output.chunk(2)
                  model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

             latents = sampler.step(timestep, latents, model_output)

             to_idle(diffusion)

             decoder = model["decoder"]
             decoder.to(decoder)

             images = decoder(latents)
             images = rescale(images, (-1,-1),(0,255),clamp=True)
             images = images.permute(0,2,3,1)

             images = images.to("cpu", torch.uint8).numpy

def rescale(x , old_range , new_range , clamp = False):
     old_min , old_max = old_range
     new_min , new_max = new_range

     x -= old_min
     x *= (new_max-new_min) / (old_max- old_min)
     x += new_min
     x += new_min

     if clamp:
          x = x.clamp(new_min, new_max)
     return x


def get_time_embedding(timestep):
     
     freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
     x = torch.tensor([timestep], dtype=torch.float32)[:None]* freqs[None]
     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

        


                  





