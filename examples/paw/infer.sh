export FAIRSEQ_PATH=$PWD/../../../fairseq
export DATAPATH=$PWD/../../../Workspace/hylab2/data/

export SAVEDIR=/home/jpong/hylab2/examples/ft_self/outputs/2023-04-09/21-03-26/checkpoints/checkpoint_last.pt
export MODEL=data2vec
export GPUS=0

for subset in test-clean test-other
do
    echo "Model is decoded on ${subset}"
    CUDA_VISIBLE_DEVICES=$GPUS python $FAIRSEQ_PATH/examples/speech_recognition/new/infer.py \
        --config-dir $FAIRSEQ_PATH/examples/speech_recognition/new/conf --config-name infer \
        task=audio_finetuning task.data=$DATAPATH/LS_p13n/$subset task.labels=ltr common.user_dir=$FAIRSEQ_PATH/examples/$MODEL \
        decoding.type=viterbi dataset.gen_subset=train \
        common_eval.path=$SAVEDIR distributed_training.distributed_world_size=1
done
