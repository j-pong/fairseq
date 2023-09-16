export FAIRSEQ_PATH=$PWD/../../../fairseq
export DATAPATH=$PWD/../../../Workspace/hylab2/data/

export SAVEDIR=/home/jpong/hylab2/examples/ft_otta/outputs/2023-05-15/17-17-33/checkpoints  
export MODEL=data2vec

files=$(ls $SAVEDIR/checkpoint.best_wer_* | sort -t _ -k 2 -n)
i=1
for file in $files; do
    if [ ! -f $SAVEDIR/checkpoint$i.pt ]; then
        ln -s $file $SAVEDIR/checkpoint$i.pt
    fi
    i=$((i+1))
done

for i in 1 3 5 10
do
    export CHECKPOINT_FILENAME=avg_${i}_best_checkpoint.pt
    python $FAIRSEQ_PATH/scripts/average_checkpoints.py \
    --inputs ${SAVEDIR} --num-epoch-checkpoints $i --checkpoint-upper-bound $i \
    --output "${SAVEDIR}/${CHECKPOINT_FILENAME}"

    for subset in test-clean test-other
    do
        echo "Model is decoded on LS/${subset}"
        export CKPTPATH=$SAVEDIR/$CHECKPOINT_FILENAME
        python $FAIRSEQ_PATH/examples/speech_recognition/new/infer.py \
            --config-dir $FAIRSEQ_PATH/examples/speech_recognition/new/conf --config-name infer \
            task=audio_finetuning task.data=$DATAPATH/LS/$subset task.labels=ltr common.user_dir=$FAIRSEQ_PATH/examples/$MODEL \
            decoding.type=viterbi dataset.gen_subset=train \
            common_eval.path=$CKPTPATH distributed_training.distributed_world_size=1
    done

    for subset in valid test
    do
        echo "Model is decoded on TED/${subset}"
        export CKPTPATH=$SAVEDIR/$CHECKPOINT_FILENAME
        python $FAIRSEQ_PATH/examples/speech_recognition/new/infer.py \
            --config-dir $FAIRSEQ_PATH/examples/speech_recognition/new/conf --config-name infer \
            task=audio_finetuning task.data=$DATAPATH/TED/ted3-total task.labels=ltr common.user_dir=$FAIRSEQ_PATH/examples/$MODEL \
            decoding.type=viterbi dataset.gen_subset=$subset \
            common_eval.path=$CKPTPATH distributed_training.distributed_world_size=1
    done
done
