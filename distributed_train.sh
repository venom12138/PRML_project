srun -p video \
     -n 1 \
     --cpus-per-task=8 \
     --gres=gpu:2 \
     --quotatype=reserved \
     python /mnt/cache/xingsen.vendor/homework/color_match/finetuning.py --batch_size 512
