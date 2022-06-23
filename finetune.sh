export WANDB_PROJECT=PRML
export exp_name=0623_single_finetune
export lr=5e-5
export run_name="lr=$lr,dirty_cn_en"
CUDA_VISIBLE_DEVICES=0 python single_finetuning.py --model ViT-B32 --batch_size 16 --lr $lr