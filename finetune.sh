export WANDB_PROJECT=PRML
export exp_name=0621_finetune
export run_name=test
CUDA_VISIBLE_DEVICES=0 python finetuning.py --model ViT-B32 --batch_size 128