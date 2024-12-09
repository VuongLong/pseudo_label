# submission

### source-combined Setting
```
CUDA_VISIBLE_DEVICES=1 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome  --output_dir 'outputs/OH/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1 --training_mode source-combined
```


### Multi-source Setting
```
CUDA_VISIBLE_DEVICES=0 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome  --output_dir 'outputs/OH/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1 --training_mode multi-source
```
