import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator : torch.Generator, num_training_steps=1000, beta_start: float= 0.00085, beta_end: float =0.0120 ):

        #betas scheduler : noise variance scheduler


        #here we are using linear scheduler but we could also use cosine scheduler or other schedulers
        self.betas = torch.linspace(beta_start ** 0.5 , beta_end ** 0.5 , num_training_steps , dtype=torch.float32) ** 2

        #we could also go from initial step to any step we want in between
        self.alpha = 1- self.betas
        self.alpha_cumprod = torch.cumprod(self.alpha , 0)  #[alpha_1 * alpha_2 * ... alpha_t ] 
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())        #we want to reverse this so from 1000 to 0 


    #training steps could be different from the inference steps
    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps// self.num_inference_steps     
        timesteps = (np.arange(0, num_inference_steps)* step_ratio).round()[::-1].copy().astype(np.int64)       #which timestep to pick from the training timesteps
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep : int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t
    
    def _get_variance(self , timestep : int) -> torch.Tensor :
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1- alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev ) / (1- alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance , min = 1e-20)

        return variance
    
    def set_strength(self , strength = 1):

        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step :]
        self.start_step = start_step
    

    def step(self , timestep: int , latents: torch.Tensor , model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1- alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # compute the predicted original sample using formula 15 of ddpm paper
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # compute the coefficient for pred_original_sample and current_sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 *  current_beta_t) / beta_prod_t
        current_sample_coeff= current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t > 0:
             device = model_output.device
             noise = torch.randn(model_output.shape , generator=self.generator , device = device , dtype = model_output.dtype)
             variance = (self._get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    # adding noise to the original samples according to the alphas and betas schedule at a given timestep(at any training step)
    def add_noise(self, original_samples:torch.FloatTensor , timesteps : torch.IntTensor) -> torch.FloatTensor:
        alpha_cumpod = self.alpha_cumprod.to(device = original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        #mean 
        sqrt_alpha_prod = alpha_cumpod[timesteps]**0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()     #alpha is a number but we need to add it to the samples which have multiple channels and batch size
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)     #adding dimension until we have the same dimension as the original samples

        #variance
        sqrt_one_minus_alpha_prod = (1- alpha_cumpod[timesteps])** 0.5      #we want standard deviation
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):

            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        #in order to add noise we need to sample some noise first 
        # here noise follows random distribution with mean 0 and variance 1
        #need to look into mmore
        noise = torch.randn(original_samples.shape, generator=self.generator , device=original_samples.device , dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise

        return noisy_samples
