python main.py \
    --data_path /home/casxm/wangjl/data/cifar-incremental \
    --num_class 30 \
    --num_task 3 \
    --dataset mycifar30 \
    --train_batch 128 \
    --seed 1 \
    --schedule 80 100 \
    --epochs 120 \
    --kd \
    --task_order 012 \
    --device 0