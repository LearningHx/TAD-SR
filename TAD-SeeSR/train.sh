

python train_tadsr.py \
--pretrained_model_name_or_path="./preset/models/stable-diffusion-2-base" \
--controlnet_model_name_or_path_Tea='./preset/seesr' \
--unet_model_name_or_path_Tea='./preset/seesr' \
--controlnet_model_name_or_path_Stu='./preset/seesr' \
--unet_model_name_or_path_Stu='./preset/seesr' \
--output_dir="./experience/tadsr" \
--root_folders './preset/dataset/train_realsr' \
--ram_ft_path './preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=2e-5 \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--max_train_steps=30000 \
--checkpointing_steps=5000