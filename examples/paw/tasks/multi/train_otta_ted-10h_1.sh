export FAIRSEQ_PATH=$PWD/../../../fairseq
export DATA_PATH=$PWD/../../../Workspace/hylab2

export MODEL=data2vec
export GPUS=1
export CONFIG=base_10h
export CKPTPATH=$PWD/../../../Workspace/hylab2/downloads/audio_base_ls.pt
export CKPTPATH2=$PWD/../../../Workspace/hylab2/downloads/audio_base_ls_100h.pt

CUDA_VISIBLE_DEVICES=$GPUS fairseq-hydra-train task.data=$DATA_PATH/data/TED/ted-10h \
    common.user_dir=$FAIRSEQ_PATH/examples/$MODEL common.tensorboard_logdir=tb \
    task.normalize=true \
    distributed_training.distributed_world_size=1 \
    optimization.lr='[0.00005]' \
    dataset.gen_subset=train dataset.valid_subset=valid dataset.max_tokens=5200000 \
    model.w2v_path=$CKPTPATH +model.s2t_path=$CKPTPATH2 model.mask_prob=0.65 \
    model.layerdrop=0.05 +model.task_specific_head=false +model.task_specific_head_temperature=0.0075 \
    +model.freeze_except_bias=true \
    +model.apply_prior_mask=false +model.freeze_prior_applies=4000 \
    optimization.max_update=20000 optimization.update_freq='[1]' \
    checkpoint.keep_best_checkpoints=10 \
    --config-name $CONFIG \
    --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning
