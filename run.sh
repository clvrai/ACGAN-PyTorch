python main.py --outf=outputs_cifar10 --niter=500 --batchSize=100 --cuda --dataset=cifar10 --imageSize=32 --dataroot=/home/telinwu/cifar10 --gpu=0
#python main.py --outf=outputs_imgnet_v1 --batchSize=100 --cuda --dataset=imagenet --niter=500 --dataroot=/home/telinwu/ImageNet/train --gpu=1 --checkpoint= 2>&1 | tee outputs_imgnet_v1/log.txt
#python main.py --outf=outputs_imgnet_10_20 --batchSize=100 --cuda --dataset=imagenet --niter=500 --dataroot=/home/telinwu/ImageNet/train --gpu=1 2>&1 | tee outputs_imgnet_10_20/log_cond.txt
