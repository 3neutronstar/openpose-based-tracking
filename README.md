# openpose-based-tracking
Openpose(CVPR 2017) and Appearance-based-Tracking(CVPR2020)

```
python main.py --load-pose --root ~/tracking_data \
 -s market1501\
 -t occlusion_reid p_duke partial_reid\
 --save-dir ./experiment/PVPM\
 -a pose_p6s --gpu-devices 0\
 --fixbase-epoch 1\
 --open-layers pose_subnet\
 --new-layers pose_subnet\
 --transforms random_flip\
 --optim sgd --lr 0.02\
 --stepsize 15 25 --staged-lr\
 --height 384 --width 128\
 --batch-size 32\
 --start-eval 1\
 --eval-freq 1\
 --load-weights ./experiment/market_PCB/model.pth.tar-60\
 --train-sampler RandomIdentitySampler\
 --reg-matching-score-epoch 0\
 --graph-matching
 --max-epoch 1
 --part-score
```
