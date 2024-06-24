# Note! 
# # For each experimental attempt, you need to change the 3 dirs below. And the rest of the parsers, except for mode, should be the same as in the train.sh file.
# python main.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 15\
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/final_DA_Strict" \
#     --log_dir "./log/final_DA_Strict" \
#     --result_dir "./result/final_DA_Strict" \
#     --mode "test"

# python main_uni.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 15\
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/final_uni_focal_Strict" \
#     --log_dir "./log/final_uni_focal_Strict" \
#     --result_dir "./result/final_uni_focal_Strict" \
#     --mode "test"


########## HIRID & PIC Dataset ##########
# python main.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_pic" \
#     --log_dir "./log/hirid_pic" \
#     --result_dir "./result/hirid_pic" \
#     --mode "test"

# python main_uni.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_picu_uni" \
#     --log_dir "./log/hirid_picu_uni" \
#     --result_dir "./result/hirid_picu_uni" \
#     --mode "test"

########## HIRID & MIMIC Split Dataset ##########
# python main.py \
#     --lr 0.001 \
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_mimic" \
#     --log_dir "./log/hirid_mimic" \
#     --result_dir "./result/hirid_mimic" \
#     --mode "test" \
#     --pic_data_dir ./dataset/MIMIC_ARDS_12H_SPLIT.csv.gz 

# python main_uni.py \
#     --lr 0.001 \
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_mimic_uni" \
#     --log_dir "./log/hirid_mimic_uni" \
#     --result_dir "./result/hirid_mimic_uni" \
#     --mode "test" \
#     --pic_data_dir "./dataset/MIMIC_ARDS_12H_SPLIT.csv.gz" 

########## HIRID & MIMIC Split Dataset Drop Derivate Variable ##########
# python main.py \
#     --lr 0.001 \
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 500 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 500\
#     --num_cont 22 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_mimic_drop" \
#     --log_dir "./log/hirid_mimic_drop" \
#     --result_dir "./result/hirid_mimic_drop" \
#     --mode "test" \
#     --pic_data_dir "./dataset/MIMIC_ARDS_12H_SPLIT.csv.gz" 

# python main_uni.py \
#     --lr 0.001 \
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 500 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 500\
#     --num_cont 22 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_mimic_uni_drop" \
#     --log_dir "./log/hirid_mimic_uni_drop" \
#     --result_dir "./result/hirid_mimic_uni_drop" \
#     --mode "test" \
#     --pic_data_dir "./dataset/MIMIC_ARDS_12H_SPLIT.csv.gz" 


########## HIRID & MIMIC All Dataset  ##########
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
    --hirid_data_dir "./Data/HiRID_ARDS_12H_SPLIT.csv.gz" \
    --mimic_data_dir "./Data/MIMIC_ARDS_12H_SPLIT.csv.gz" 

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
#     --hirid_data_dir "./Data/HiRID_ARDS_12H_SPLIT.csv.gz" \
#     --pic_data_dir "./Data/MIMIC_ARDS_12H_SPLIT.csv.gz" 