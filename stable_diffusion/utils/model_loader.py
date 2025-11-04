from stable_diffusion.models.clip import Clip
from stable_diffusion.models.vae.encoder import VAE_Encoder
from stable_diffusion.models.vae.decoder import VAE_Decoder
from stable_diffusion.models.diffusion.diffusion_unet import Diffusion
from .model_converter import load_from_standard_weights
import torch

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)  # Fixed: use VAE_Decoder
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = Clip().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)


    return {
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
        "clip": clip
    }

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    ckpt_path = "../dataa/v1-5-pruned-emaonly.ckpt"
    models = preload_models_from_standard_weights(ckpt_path, device)
    print("Models loaded successfully.")