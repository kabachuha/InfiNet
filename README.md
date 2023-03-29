# InfiNet

*WIP, tests are to be done!*

InfiNet — ControlNet-like structure for video diffusion (Unet3D-based) models allowing them to train on arbitrary long videos and as result produce extremely long videos on consumer PCs via the DiffusionOverDiffusion architecture proposed by Microsoft in https://arxiv.org/abs/2303.12346 for their NUWA-XL model.

Thanks to it utilizing so-called *zero-convolutions*, it's possible to add the InfiNet model on top of an already pretrained U-net to save resources.

This repo contains the code of ModelScope's text2video model with InfiNet being injected into it directly. After I test whether it works, I'll try to figure out a more elegant way to hijack it.

The InfiNet module itself is located here https://github.com/kabachuha/InfiNet/blob/master/t2v_modules/dod_unet.py

### References

1. StabilityAI's Stable Diffusion https://github.com/CompVis/stable-diffusion
2. Microsoft's NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation https://arxiv.org/abs/2303.12346
3. lllyasviel's ControlNet https://github.com/lllyasviel/ControlNet
