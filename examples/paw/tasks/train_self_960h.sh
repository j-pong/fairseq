# data2vec
export FAIRSEQ_PATH=$PWD/../../../fairseq
export DATA_PATH=$PWD/../../../Workspace/hylab2

export MODEL=data2vec
export GPUS=0,1,2,3
export CONFIG=base_960h
export CKPTPATH=$PWD/../../../Workspace/hylab2/downloads/audio_base_ls.pt

CUDA_VISIBLE_DEVICES=$GPUS fairseq-hydra-train task.data=$DATA_PATH/data/LS/train-960 \
    common.user_dir=$FAIRSEQ_PATH/examples/$MODEL common.tensorboard_logdir=tb \
    task.normalize=true \
    distributed_training.distributed_world_size=4 \
    optimization.lr='[0.00005]' \
    dataset.gen_subset=train dataset.valid_subset=valid dataset.max_tokens=5200000 \
    model.w2v_path=$CKPTPATH model.mask_prob=0.0 +model.apply_prior_mask=false +model.freeze_prior_applies=40000 \
    optimization.max_update=400000 \
    checkpoint.keep_best_checkpoints=10 \
    --config-name $CONFIG \
    --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning
