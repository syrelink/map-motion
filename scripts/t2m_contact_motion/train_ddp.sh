EXP_NAME=$1
PORT=$2

if [ -z "$PORT" ]
then
    PORT=29500
fi

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} train_ddp.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            diffusion.steps=1000 \
            task=text_to_motion_contact_motion_gen \
            task.dataset.sigma=0.8 \
            task.train.batch_size=64 \
            task.train.max_steps=600000 \
            task.train.save_every_step=100000 \
            task.dataset.train_transforms=['RandomEraseLang','RandomEraseContact','NumpyToTensor'] \
            model=cmdm \
            model.arch='trans_enc' \
            model.data_repr='h3d' \
            model.text_model.max_length=20
