## WaveCastNet Training
### Dense and Sparse Sampling Scenarios
```
python WaveCastNet/earthquake_train.py --model LEM_dense --num_kernels 144 --activation tanh --batch_size 64 --learning_rate 5e-4
python WaveCastNet/earthquake_train.py --model LEM_sparse --num_kernels 144 --activation tanh --batch_size 64 --learning_rate 5e-4
```
### Uncertainty Quantification Training
```
python /global/homes/d/dwlyu/WaveCastNet/earthquake_train.py --model LEM_dense --num_kernels 144 --activation tanh --batch_size 64 --learning_rate 5e-4 --training_uq 1 --load_seed 2
```
### Seq2Seq Ablation Studies
```
python WaveCastNet/earthquake_train.py --model LEM --num_kernels 144 --activation tanh --batch_size 64 --learning_rate 5e-4
python WaveCastNet/earthquake_train.py --model LSTM --num_kernels 144 --activation tanh --batch_size 64 --learning_rate 5e-4
python WaveCastNet/earthquake_train.py --model GRU --num_kernels 144 --activation tanh --batch_size 64 --learning_rate 5e-4
```
### Comparative Studies with Transformer Architectures
```
python WaveCastNet/earthquake_train.py --model Swin --num_kernels 144 --patch_size 3 4 4 --batch_size 64 --learning_rate 5e-4
python WaveCastNet/earthquake_train.py --model Time-s-pyramid --num_kernels 192 --patch_size 1 8 8 --batch_size 64 --learning_rate 5e-4
python WaveCastNet/earthquake_train.py --model Time-s-plain --num_kernels 192 --patch_size 1 8 8 --batch_size 64 --learning_rate 5e-4
```
# Experiments in Appendix

## Train Moving MNIST

```
export CUDA_VISIBLE_DEVICES=4; python MovingMnist_train.py --model LEM -lr 5e-4 --num_layers 3 --width 64
export CUDA_VISIBLE_DEVICES=4; python MovingMnist_train.py --model LSTM -lr 5e-4 --num_layers 3 --width 64
export CUDA_VISIBLE_DEVICES=4; python MovingMnist_train.py --model QRNN -lr 5e-4 --num_layers 3 --width 64
```

## Train RBC Fluid Flow
```
python RBC_train.py --model LEM -lr 5e-4 --width 72 --activation tanh --input_steps 50 --future_steps 50
python RBC_train.py --model LSTM -lr 5e-4 --width 72 --activation tanh --input_steps 50 --future_steps 50
python RBC_train.py --model QRNN -lr 5e-4 --width 72 --activation tanh --input_steps 50 --future_steps 50
```