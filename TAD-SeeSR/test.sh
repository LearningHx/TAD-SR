python test_tadsr.py \
--pretrained_model_path ./preset/models/stable-diffusion-2-base \
--prompt '' \
--tadsr_model_path./TAD-SeeSR/preset/tadsr \
--ram_ft_path ./preset/models/DAPE.pth \
--image_path ./dataset/RealSR_CenterCrop-20240520T083645Z-001/RealSR_CenterCrop/test_LR \
--output_dir ./TAD-SeeSR/results \
--start_point lr \
--num_inference_steps 1 
