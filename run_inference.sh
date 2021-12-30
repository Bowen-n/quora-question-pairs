python inference.py \
    --mode all \
    --model_path qqp_roberta-base_ce/pytorch_model.bin \
    --output_dir inference_base_ce && \

python inference.py \
    --mode all \
    --model_path qqp_roberta-base_pretrained_e10_ce/pytorch_model.bin \
    --output_dir inference_pre_ce && \

python inference.py \
    --mode all \
    --model_path qqp_roberta-base_pretrained_e10_focal/pytorch_model.bin \
    --output_dir inference_pre_focal
