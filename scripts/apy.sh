cd ..

# APY w/o finetuning GZSL

python train2.py --dataset APY  --ga 0.1 --beta 0.2 --dis 1 --nSample 1200 --gpu 0 --S_dim 128 --NS_dim 128 \
  --lr 0.0001  --classifier_lr 0.005 --gen_nepoch 400 --kl_warmup 0.02 --tc_warmup 0.001 --weight_decay 1e-6 \
  --vae_enc_drop 0.4 --vae_dec_drop 0.5 --dis_step 2 --ae_drop 0.2 --evl_start 15000 --evl_interval 200 \
  --classifier_steps 20 --manualSeed 3861 \
  --center_margin 30 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.5 \
  --attSize 64 --nclass_seen 20 --ngh 1024 --CS_dim 128 --CNS_dim 128
# APY w/o finetuning ZSL
#python train.py --dataset APY  --ga 0.1 --beta 1 --dis 1 --nSample 1200 --gpu 0 --S_dim 128 --NS_dim 128 \
#--lr 0.0001  --classifier_lr 0.01 --gen_nepoch 400 --kl_warmup 0.02 --tc_warmup 0.001 --weight_decay 1e-6 \
#--vae_enc_drop 0.4 --vae_dec_drop 0.5 --dis_step 2 --ae_drop 0.2 --evl_start 15000 --evl_interval 200 \
#--classifier_steps 20 --manualSeed 3861 --zsl true #
#
## APY w/ finetuning GZSL
#python train.py --dataset APY  --ga 0.3 --beta 1 --dis 0.3 --nSample 1200 --gpu 1 --S_dim 1024 --NS_dim 1024 \
#  --lr 0.0001  --classifier_lr 0.001 --gen_nepoch 250 --kl_warmup 0.001 --tc_warmup 0.0003 --weight_decay 3e-6 \
#  --vae_enc_drop 0.4 --vae_dec_drop 0.5 --dis_step 2 --ae_drop 0.2 --evl_start 0 --evl_interval 200 \
#  --classifier_steps 20 --finetune true --manualSeed 3861 #
#
## APY w/ finetuning GZSL
#python train.py --dataset APY  --ga 0.3 --beta 1 --dis 1 --nSample 1200 --gpu 1 --S_dim 128 --NS_dim 128 \
#  --lr 0.00001  --classifier_lr 0.004 --gen_nepoch 400 --kl_warmup 0.02 --tc_warmup 0.001 --weight_decay 3e-6 \
#  --vae_enc_drop 0.2 --vae_dec_drop 0.5 --dis_step 1 --ae_drop 0.2 --evl_start 0 --evl_interval 200 \
#  --finetune true --manualSeed 3861 --zsl true #


