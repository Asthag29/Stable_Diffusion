import torch
import numpy as np
from tqdm import tqdm
from .ddpm import DDPMSampler
# import model_loader
from PIL import Image
from transformers import CLIPTokenizer
import torch

from .utils.model_loader import preload_models_from_standard_weights
from .models.attention import SelfAttention

Width = 512     #final image width(input)
Height = 512    #final image height(inpout)
Latent_Width = Width // 8       #latent image width(vae encoder output)
Latent_Height = Height // 8        #latent image height(vae encoder output)

# more noise more creative, less noise less creative

# cfg_scale = 7.5 (difference between conditioned and unconditioned)
# uncon_prompt = without any condition
# each time step indicates the noise level              #python linting
#higher strength means higher inference steps which means less attention to starting image

def generate(
    prompt: str,    #what we want to generate
    uncon_prompt: str,  #negative prompt for classifier free guidance or an empty string
    input_image=None,   #input image for modification
    strength=0.8,   #how much attention do we want to pay to starting image(influence)
    do_cfg=True,        #whether to use classifier free guidance or not
    sampler_name="ddpm",
    n_inference_steps=50,   #how many inference steps we want to do
    model={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    cfg_scale=7.5,      #how much value we want our model to give to prompt(w) in the baove image
    video: bool = False,
):
    
    with torch.no_grad():
        if not (0 < strength <= 1): 
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)       #to_idle is now a function which takes x as input and returns x.to(idle_device) as output
        else:
            to_idle = lambda x: x

        generate = torch.Generator(device=device)   #random number generator for noise generation
        if seed is None:
            generate.seed()
        else:
            generate.manual_seed(seed)

        clip = model["clip"]        #how small clip
        clip.to(device)

        #classifier free guidance is text to image generattion
        if do_cfg:  #if we want to do classifier free guidance  we can decide how much the model want to pay attention to the prompt
            # convert the prompt into tokens using tokenizer

            # convert the tokens into embedding and if the prompt is too short we will add padding of max_length which i need to go deep in
            cond_tokens = tokenizer.batch_encode_plus( [prompt], padding="max_length", max_length=77 ).input_ids        #max_length is the number of tokens it could process at once(if its small it would be padded) and then we take the input ids from the dictionaryf

            #converting it into tensor (batch_size , seq_len/context length)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            #input ids will be converted into embedding using clip model
            cond_context = clip(cond_tokens)        #(batch_size , seq_len/context length) -> (batch_size , seq_len/context length , dim/channels/embedding = 768)


            #same for unconditional output what would be the output if there is no prompt can be empty string also
            uncon_tokens = tokenizer.batch_encode_plus([uncon_prompt], padding="max_length", max_length=77).input_ids
            uncon_tokens = torch.tensor(uncon_tokens, dtype=torch.long, device=device)
            uncon_context = clip(uncon_tokens)

            # concatenating both the context vectors
            context = torch.cat([cond_context, uncon_context])  #(batch_size*2 , seq_len/context length , dim/channels/embedding )
            # print(f"context_max: {context.max().item():.3f}, context_min: {context.min().item():.3f}")

        else:       #if we don't want to do classifier free guidance, but we cannot decide here how much attention to pay to the prompt
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77 ).input_ids
            tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            context = clip(tokens)      #(batch_size , seq_len/context length) -> (batch_size , seq_len/context length , dim/channels/embedding )
            # print(f"context_max: {context.max().item():.3f}, context_min: {context.min().item():.3f}")
        to_idle(clip)  # for moving the model to the cpu after using them 

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generate)
            sampler.set_inference_timesteps(n_inference_steps)  
        else:
            raise ValueError(f"sampler {sampler_name} not recognized")

        latents_shape = (1, 4, Latent_Height, Latent_Width)     #(batch_size , 4 , height/4 , width/4)

        if input_image: #if we have a input image pass it throught the vae(but why, mayeb because we want to generte the similar structure )
            encoder = model["encoder"]
            encoder.to(device)

           # few preprocessing steps for input image(would be better if we could make a separate function for this)
            input_image_tensor = input_image.resize((Height, Width))    #shape is ( , height , width, channl)
            input_image_tensor = np.array(input_image_tensor)   # array ( height , width , channel )
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)  #(height , width , channel )

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1)) #rescaling 
            input_image_tensor = input_image_tensor.unsqueeze(0)    # adding the batch dimension (1, height , width , channel )
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2) #changing order to tensor (1, channel , height , width)

            # generating the noise of the latent shape 
            encoder_noise = torch.randn(latents_shape, generator=generate, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            #need to work thorught this function
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])  #also i think the noise is not concatented but rather added becuase th eunet cannot handle 2 batches (one for latent and one for noise)

            to_idle(encoder)

        else:   # if we dont have a input image then a random generator will randomly generate the latent shape noise, it need not to be passed through the encoder
            latents = torch.randn(latents_shape, generator=generate, device=device)

        # print(f"Initial latents: min={latents.min():.3f}, max={latents.max():.3f}")

        diffusion = model["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)     #how many timesteps we want to do

       
        latents_list = []  # for video generation
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if do_cfg:  #doubling becuase we have two context vectors(one for conditioned and one for unconditioned)
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)      #time_embedding is flexible it works according to batch_size
            # why do we predict noise instead of the denoised image itself

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)      #double the batch size of the usual ones    #first output is context second is without context
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            latents = sampler.step(timestep, latents, model_output)     #need to understand this function
            if video:
                latents_list.append(latents.cpu().clone())
        

        to_idle(diffusion)

        decoder = model["decoder"]
        decoder.to(device)

        if video:
            images = []
            for latents in latents_list:
                latents = latents.to(device)
                image = decoder(latents)
                image = rescale(image, (-1, 1), (0, 255), clamp=True)
                image = image.permute(0, 2, 3, 1)
                image = image.to("cpu", torch.uint8).numpy()
                images.append(image[0])   #appending the first image in the batch
            to_idle(decoder)
            return images   #returning all the images for video generation

        else:
            images = decoder(latents)
            to_idle(decoder)

            images = rescale(images, (-1, 1), (0, 255), clamp=True)
            images = images.permute(0, 2, 3, 1)
            images = images.to("cpu", torch.uint8).numpy()

            return images[0]   #returning the first image in the batch

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):       #need to understand this function
    #(160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    #(1,160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    #(1,160*2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)



