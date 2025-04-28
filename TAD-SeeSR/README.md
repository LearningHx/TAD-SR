

## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/LearningHx/TAD-SR.git
cd TAD-SR

# create an environment with python >= 3.8
conda create -n tadsr python=3.8
conda activate tadsr
pip install -r requirements.txt
```

## 🚀 Fast Testing
#### Step 1: Download the pretrained models
- Download the pretrained SD-2-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base).
- Download the TAD-SR models from [GoogleDrive](https://drive.google.com//drive/folders/13VcWNQjRcLK94ijYkxrED-k0_32ec3B0?hl=zh-TW)
- Download the DAPE models from [GoogleDrive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link)

You can put the models into `preset/`.

#### Step 2: Prepare testing data
You can put the testing images in the `preset/datasets/test_datasets`.

#### Step 3: Running testing command
```
python test_tadsr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt '' \
--tadsr_model_path preset/models/tadsr \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/datasets/test_datasets \
--output_dir preset/datasets/output \
--start_point lr \
--num_inference_steps 1
```


#### Test Benchmark
`RealLR200`, `RealSR` and `DRealSR` can be downloaded from [SeeSR](https://drive.google.com/drive/folders/1L2VsQYQRKhWJxe6yWZU9FgBWSgBCk6mz?usp=drive_link).

## 🌈 Train 

#### Step1: Download the pretrained models
Download the pretrained [SD-2-base models](https://huggingface.co/stabilityai/stable-diffusion-2-base), [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth), [SeeSR](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link). You can put them into `preset/models`.

#### Step2: Prepare training data
We employ the same preprocessing measures as SeeSR. To further accelerate the model's training speed, we pre-generated corresponding super-resolution images for the low-resolution images using the SeeSR model and stored them in `root_folders/tch`.

#### Step3: Training for TAD-SR
```
 CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7" accelerate launch train_tadsr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--controlnet_model_name_or_path_Tea='preset/seesr' \
--unet_model_name_or_path_Tea='preset/seesr' \
--controlnet_model_name_or_path_Stu='preset/seesr' \
--unet_model_name_or_path_Stu='preset/seesr' \
--output_dir="./experience/tadsr" \
--root_folders 'DataSet/training' \
--ram_ft_path 'preset/models/DAPE.pth' \
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
```
- `--pretrained_model_name_or_path` the path of pretrained SD model from Step 1
- `--root_folders` the path of your training datasets from Step 2
- `--ram_ft_path` the path of your DAPE model from Step 3



