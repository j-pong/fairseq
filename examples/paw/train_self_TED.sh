# data2vec
export FAIRSEQ_PATH=$PWD/../../../fairseq
export DATA_PATH=$PWD/manifest/TED

export MODEL=data2vec
export GPUS=0,1,2,3
export CONFIG=base_960h
export CKPTPATH=$PWD/../../../Workspace/hylab2/downloads/audio_base_ls.pt

CUDA_VISIBLE_DEVICES=$GPUS fairseq-hydra-train task.data=$DATA_PATH/ted3-total-d2v_100h_state \
    common.user_dir=$FAIRSEQ_PATH/examples/$MODEL common.tensorboard_logdir=tb \
    task.normalize=true \
    distributed_training.distributed_world_size=4 \
    optimization.lr='[0.00004]' \
    dataset.gen_subset=train dataset.valid_subset=valid dataset.max_tokens=5200000 \
    model.w2v_path=$CKPTPATH model.mask_prob=0.6 +model.apply_prior_mask=true +model.freeze_prior_applies=40000 \
    optimization.max_update=200000 \
    checkpoint.keep_best_checkpoints=10 \
    --config-name $CONFIG \
    --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning

# data2vec2
# export FAIRSEQ_PATH=$PWD/../../../fairseq
# export DATA_PATH=$PWD/../../../Workspace/hylab2

# export MODEL=data2vec
# export GPUS=0,1,2,3
# export CONFIG=base_100h
# export CKPTPATH=$PWD/../../../Workspace/hylab2/downloads/base_libri.pt

# CUDA_VISIBLE_DEVICES=$GPUS fairseq-hydra-train task.data=$DATA_PATH/data/LS/train-clean-100 \
#     common.user_dir=$FAIRSEQ_PATH/examples/$MODEL common.tensorboard_logdir=tb \
#     task.normalize=true \
#     distributed_training.distributed_world_size=4 \
#     dataset.gen_subset=train dataset.gen_subset=train dataset.valid_subset=valid dataset.max_tokens=5200000 \
#     model.w2v_path=$CKPTPATH +model.apply_prior_mask=false \
#     optimization.update_freq='[1]' \
#     checkpoint.keep_best_checkpoints=10 \
#     --config-name $CONFIG \
#     --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning

# export FAIRSEQ_PATH=$PWD/../../../fairseq
# export DATA_PATH=$PWD/../../../Workspace/hylab2

# export MODEL=data2vec
# export GPUS=0,1,2,3
# export CONFIG=base_10h
# export CKPTPATH=""

# CUDA_VISIBLE_DEVICES=$GPUS fairseq-hydra-train task.data=$DATA_PATH/data/TED/ted-10h-d2v2_100h_state@2 \
#     common.user_dir=$FAIRSEQ_PATH/examples/$MODEL common.tensorboard_logdir=tb \
#     task.normalize=true \
#     distributed_training.distributed_world_size=4 \
#     optimization.lr='[0.00005]' \
#     dataset.gen_subset=train dataset.valid_subset=valid dataset.max_tokens=5200000 \
#     model.w2v_path=$CKPTPATH model.mask_prob=0.65 \
#     optimization.max_update=20000 optimization.update_freq='[2]' \
#     checkpoint.keep_best_checkpoints=10 \
#     --config-name $CONFIG \
#     --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning

