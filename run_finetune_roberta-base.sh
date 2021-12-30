python finetune.py \
    --pre_name roberta-base \
    --loss_type ce \
    --num_epochs 15 \
    --batch_size 30 \
    --num_workers 26 \
    --output_dir qqp_roberta-base_ce \
    --use_fp16
