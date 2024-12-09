# submission


```
CUDA_VISIBLE_DEVICES=1 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset OfficeHome  --output_dir 'outputs/OH1/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --M1 16 --M2 16 --data_root /mnt/SSD1/datasets/DG_data/DomainBed/ --dataset ImageCLEF  --output_dir 'outputs/IC/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1
```
