export WANDB_PROJECT=PRML
export exp_name=0622_finetune
export run_name=test
CUDA_VISIBLE_DEVICES=0 python single_finetuning.py --model ViT-B32 --batch_size 16