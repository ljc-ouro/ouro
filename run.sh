nohup python -m torch.distributed.run --nproc_per_node=1 main.py > log/train.log 2>&1 &
# python -m torch.distributed.run --nproc_per_node=1 main.py
# tensorboard --logdir=log