cd pytorch-train
nohup python train.py --cuda --adadelta >/tmp/crnnlog10.log 2>&1 &
