# python train.py -a vgg16 --dist-url 'tcp://127.0.0.1:5555' --dist-backend 'nccl' \
#     --multiprocessing-distributed --world-size 1 --rank 0 --pretrained \
#     --epochs 10 --prune magnitude --sparse_ratio 0.4 --print-freq 1000 --lr 0.0005 \
#     /datasets/imagenet

# python train.py -a vgg16 --pretrained \
#     --epochs 10 --prune magnitude --sparse_ratio 0.4 --print-freq 1000 --lr 0.0005 \
#     /datasets/imagenet

python train.py -a vgg16 --dist-url 'tcp://127.0.0.1:7777' --dist-backend 'nccl' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --epochs 15 --pretrained --evaluate --prune magnitude --sparse_ratio 0.4 \
    /datasets/imagenet

# python train.py -a vgg16 --epochs 10 /datasets/imagenet --prune magnitude --sparse_ratio 0.25 --pretrained
# python train.py -a vgg16 --epochs 10 /datasets/imagenet