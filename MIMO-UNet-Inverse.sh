CUDA_VISIBLE_DEVICES=2 \
python main.py \
--model_name "MIMO-UNet-SNN-invermobile" \
--mode "test" \
--device "2" \
--batch_size 2 \
--num_epoch 600 \
--save_freq 1 \
--valid_freq 1 \
--print_freq 100 \
--exp_name "inv_res1_lr3_bs2_t5" \
--data_dir "/sqy/CamVid" \
--resume "results/MIMO-UNet-SNN-invermobile/inv_res1_lr3_bs2_t5/weights/epoch_122_iou_0.565.pkl" \
--attention_map_dir "./mobile_res1_lr3_bs2_spike_map"
