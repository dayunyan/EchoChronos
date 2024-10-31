export CUDA_VISIBLE_DEVICES=0,1
export CUDA_NUM=2

mpirun -n ${CUDA_NUM} python train_simple.py