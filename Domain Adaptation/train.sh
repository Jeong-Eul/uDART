# Note! 
# You will need to change the 3 dirs below for each experiment attempt.

######### HIRID & MIMIC All Dataset  ##########
python main.py \
    --lr 0.001 \
    --weight_decay 0.00001 \
    --batch_size 256 \
    --num_epoch 100 \
    --T_max 25 \
    --scheduler \
    --early_stop \
    --patience 20\
    --num_cont 49 \
    --num_cat 9 \
    --dim 32 \
    --dim_head 16 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint/hirid_mimic_all" \
    --log_dir "./log/hirid_mimic_all" \
    --result_dir "./result/hirid_mimic_all" \
    --mode "test" \
    --hirid_data_dir "./Data/HIRID_ARDS_12H.csv.gz" \
    --mimic_data_dir "./Data/MIMIC_ARDS_12H.csv.gz" 

# python main_uni.py \
#     --lr 0.001 \
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 20\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_mimic_uni_all" \
#     --log_dir "./log/hirid_mimic_uni_all" \
#     --result_dir "./result/hirid_mimic_uni_all" \
#     --mode "test" \
#     --hirid_data_dir "./Data/HIRID_ARDS_12H.csv.gz" \
#     --pic_data_dir "./Data/MIMIC_ARDS_12H.csv.gz" 
