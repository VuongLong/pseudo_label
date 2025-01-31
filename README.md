# submission

### source-combined Setting
```
CUDA_VISIBLE_DEVICES=1 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome  --output_dir 'outputs/OH/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1 --training_mode source-combined
```


### Multi-source Setting
```
CUDA_VISIBLE_DEVICES=0 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome  --output_dir 'outputs/OH/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1 --training_mode multi-source
```


### ViT
```
CUDA_VISIBLE_DEVICES=0 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset DomainNet  --backbone ViT-L/14 --output_dir 'outputs/DN/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1 --training_mode multi-source
```

```
CUDA_VISIBLE_DEVICES=1 python main_PGA.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome  --backbone ViT-L/14
```

### Corruption

gaussian_noise, shot_noise, impulse_noise, defocus_blur,
glass_blur,  zoom_blur, snow, frost, fog,
brightness, contrast, elastic_transform, pixelate, jpeg_compression,
speckle_noise, gaussian_blur, spatter, saturate

```
CUDA_VISIBLE_DEVICES=1 python main_corruptions.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/  --dataset OfficeHome --corruption spatter --severity 1 --output_dir 'outputs_corrupted/OH/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1
```

```
CUDA_VISIBLE_DEVICES=0 python main_PGA_corruptions.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome --corruption defocus_blur --severity 1
```