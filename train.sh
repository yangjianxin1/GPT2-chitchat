export DLWS_NUM_WORKER=1
export DLWS_NUM_GPU_PER_WORKER=8
config_json=deepspeed_config/config.json
LR=1e-4
BS=88
EPOCHS=10
model_config=config/config.json
gpt_options="--lr $LR \
             --device 0,1,2,3,4,5,6,7 \
             --batch_size $BS --epochs $EPOCHS \
             --model_config $model_config"
deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
            "

full_options="${gpt_options} ${deepspeed_options}"
run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} train.py $@ ${full_options}"
echo $run_cmd
eval $run_cmd