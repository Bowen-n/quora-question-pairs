python inference.py \
    --mode test \
    --model_path qqp_roberta-base_pretrained_e10_ce/pytorch_model.bin \
    --output_dir inference_ce && \

python inference.py \
    --mode test \
    --model_path qqp_roberta-base_pretrained_e10_focal/pytorch_model.bin \
    --output_dir inference_focal && \

python stacking.py \
    --model_path stacking/linear.model \
    --test
    