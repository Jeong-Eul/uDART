# Note! 

########## HIRID & MIMIC All Dataset  ##########
python get_attmap.py \
    --batch_size 256 \
    --num_cont 49 \
    --num_cat 9 \
    --dim 32 \
    --dim_head 16 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint/hirid_mimic_all" \
    --result_dir "./result/hirid_mimic_all" \
    --mode "attn" \
    --hirid_data_dir "./Data/HIRID_ARDS_12H.csv.gz" \
    --pic_data_dir "./Data/MIMIC_ARDS_12H.csv.gz" 
