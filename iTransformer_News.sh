export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path /home/userroot/dev/iTransformer/data_prep/ \
  --data_path index_300.embedding.csv \
  --model_id EMB_16_2 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 16 \
  --pred_len 2 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --embed fixed
