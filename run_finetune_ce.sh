python finetune.py \
    --pre_name roberta-base \
    --pretrained_model_dir ./pretrain_roberta_e10 \
    --loss_type ce \
    --num_epochs 15 \
    --batch_size 30 \
    --num_workers 24 \
    --output_dir qqp_roberta-base_pretrained_e10_ce \
    --use_fp16
