cd ..

# AWA2 w/o finetuning GZSL
python train2.py --dataset AwA2 --ga 0.7 --beta 0.5 --dis 0.5 --nSample 5000 --gpu 0 --S_dim 1024 --NS_dim 1024 --lr 0.00003 \
  --classifier_lr 0.003 --kl_warmup 0.01 --tc_warmup 0.001 --vae_dec_drop 0.5 --vae_enc_drop 0.4 --dis_step 2 \
  --ae_drop 0.2 --gen_nepoch 220 --evl_start 40000 --evl_interval 400 --manualSeed 6152 \
  --center_margin 50 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.5 \
  --attSize 85 --nclass_seen 40 --ngh 4096 --CS_dim 1024 --CNS_dim 1024

## AWA2 w/o finetuning ZSL
#python train.py --dataset AWA2 --ga 0.5 --beta 1 --dis 0.3 --nSample 5000 --gpu 1 --S_dim 1024 --NS_dim 1024 --lr 0.00003 \
#  --classifier_lr 0.003 --kl_warmup 0.01 --tc_warmup 0.001 --vae_dec_drop 0.5 --vae_enc_drop 0.4 --dis_step 2 \
#  --ae_drop 0.2 --gen_nepoch 220 --evl_start 40000 --evl_interval 400 --manualSeed 6152 --zsl true
#
## AWA2 w/ finetuning GZSL
#python train2.py --dataset AwA2 --ga 0.7 --beta 0.9 --dis 0.5 --nSample 1800 --gpu 0 --S_dim 2048 --NS_dim 2048 --lr 0.00003 \
#--classifier_lr 0.0015 --kl_warmup 0.01 --tc_warmup 0.001 --vae_dec_drop 0.5 --vae_enc_drop 0.4 --dis_step 2 \
#--ae_drop 0.2 --gen_nepoch 150 --evl_start 20000 --evl_interval 300 --manualSeed 6152 --finetune true \
#--weight_decay 3e-7 --classifier_steps 20 --CS_dim 1024 --CNS_dim 1024
#
## AWA2 w/ finetuning ZSL
#python train.py --dataset AWA2 --ga 0.5 --beta 1 --dis 0.3 --nSample 1800 --gpu 0 --S_dim 312 --NS_dim 312 --lr 0.00003 \
#--classifier_lr 0.0015 --kl_warmup 0.01 --tc_warmup 0.001 --vae_dec_drop 0.5 --vae_enc_drop 0.4 --dis_step 2 \
#--ae_drop 0.2 --gen_nepoch 150 --evl_start 20000 --evl_interval 300 --manualSeed 6152 --finetune true \
#--weight_decay 3e-7 --classifier_steps 20 --zsl true #
