# data2vec test
export FAIRSEQ_PATH=$PWD/../../../fairseq
export DATA_PATH=$PWD/../../../Workspace/hylab2

export MODEL=data2vec
export GPUS=0
export CONFIG=base_960h
export CKPTPATH=$PWD/../../../Workspace/hylab2/downloads/audio_base_ls.pt

CUDA_VISIBLE_DEVICES=$GPUS fairseq-hydra-train task.data=$DATA_PATH/data/LS/train-960-d2v_100h_state common.user_dir=$FAIRSEQ_PATH/examples/$MODEL \
    task.normalize=true \
    distributed_training.distributed_world_size=1 optimization.lr='[0.00005]' \
    dataset.gen_subset=train dataset.valid_subset=valid dataset.max_tokens=5200000 \
    model.w2v_path=$CKPTPATH model.mask_prob=0.45 \
    optimization.max_update=400000 \
    --config-name $CONFIG \
    --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning