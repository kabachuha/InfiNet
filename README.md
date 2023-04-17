# InfiNet

*WIP, tests are to be done!*

InfiNet — ControlNet-like structure for video diffusion (Unet3D-based) models allowing them to train on arbitrary long videos and as result produce extremely long videos on consumer PCs via the DiffusionOverDiffusion architecture proposed by Microsoft in https://arxiv.org/abs/2303.12346 for their NUWA-XL model.

![image](https://user-images.githubusercontent.com/14872007/232623882-cec7fe8a-bfb1-4230-bd7a-fe3491388ce6.png)

Thanks to it utilizing so-called *zero-convolutions*, it's possible to add the InfiNet model on top of an already pretrained U-net to save resources.

![image](https://user-images.githubusercontent.com/14872007/232623925-c57f359b-c490-4aa6-8499-aacb87f07664.png)

This repo contains the code of ModelScope's text2video model with InfiNet being injected into it by appending it as a submodule and hijacking the `forward` function.

![image](https://user-images.githubusercontent.com/14872007/232623946-b9163777-778f-4e37-a42e-c15f536229ee.png)

The InfiNet module itself is located here https://github.com/kabachuha/InfiNet/blob/master/t2v_modules/dod_unet.py

The key difference with ControlNet is that this model has to control both Upsample and Downsample blocks, whereas ControlNet controls only the Upsample blocks, so it couldn't be just another fine-tuned CN.

### References

1. StabilityAI's Stable Diffusion https://github.com/CompVis/stable-diffusion
2. Microsoft's NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation https://arxiv.org/abs/2303.12346
3. lllyasviel's ControlNet https://github.com/lllyasviel/ControlNet

## Making Dataset for DiffusionOverDiffusion

1. Chop the large video into smaller subdivisions by launching `python video_chop.py your_vide.mp4 --L sample_frames` where `sample_frames` is the number of divisions on each level. Defaults to 12.

2. 
