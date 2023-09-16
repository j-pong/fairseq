# NOTE: Before runs these code, change (foced or pseudo) and prop_raw options 
export FAIRSEQ_PATH=$PWD/../../../fairseq
export DATA_PATH=$PWD/../../../Workspace/hylab2

export MODELTYPE=data2vec
export CKPTPATH=$PWD/../../../Workspace/hylab2/downloads/audio_base_ls_100h.pt
# export CKPTPATH=/home/jpong/hylab2/examples/ft_self/outputs/2023-05-02/14-24-49/checkpoints/avg_1_best_checkpoint.pt

python generate_state/main.py \
    --config-dir $FAIRSEQ_PATH/examples/speech_recognition/new/conf --config-name infer \
    task=audio_finetuning task.data=$DATA_PATH/data/TED/ted task.labels=ltr common.user_dir=$FAIRSEQ_PATH/examples/$MODELTYPE \
    decoding.type=viterbi dataset.gen_subset=train dataset.batch_size=1 \
    common_eval.path=$CKPTPATH distributed_training.distributed_world_size=1
