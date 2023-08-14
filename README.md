# Forecasting

## Train Moving MNIST

```
export CUDA_VISIBLE_DEVICES=4; python MovingMnist_train.py --model LEM -lr 5e-4 --num_layers 3 --width 64
export CUDA_VISIBLE_DEVICES=4; python MovingMnist_train.py --model LSTM -lr 5e-4 --num_layers 3 --width 64
export CUDA_VISIBLE_DEVICES=4; python MovingMnist_train.py --model QRNN -lr 5e-4 --num_layers 3 --width 64
```


