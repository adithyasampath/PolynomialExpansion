CUDA_VISIBLE_DEVICES=0 nohup python train.py --best_epoch=50 --hid_dim=256 --enc_pf_dim=256 --dec_pf_dim=256 --exp_name=nlayers3hdim256 > nlayers3hdim256_test.out &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --best_epoch=40 --hid_dim=512 --enc_pf_dim=512 --dec_pf_dim=512 --exp_name=nlayers3hdim512 > nlayers3hdim512_test.out &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --best_epoch=40 --hid_dim=256 --enc_pf_dim=512 --dec_pf_dim=512 --exp_name=nlayers3hdim256enc512 > nlayers3hdim256enc512_test.out &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --best_epoch=50 --hid_dim=512 --enc_pf_dim=256 --dec_pf_dim=256 --exp_name=nlayers3hdim512enc256 > nlayers3hdim512enc256_test.out &

CUDA_VISIBLE_DEVICES=0 nohup python train.py --best_epoch=45 --hid_dim=256 --enc_pf_dim=256 --dec_pf_dim=256 --enc_layers=1 --dec_layers=1 --exp_name=nlayers1hdim256 > nlayers1hdim256_test.out &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --best_epoch=50 --hid_dim=256 --enc_pf_dim=256 --dec_pf_dim=256 --enc_layers=2 --dec_layers=2 --exp_name=nlayers2hdim256 > nlayers2hdim256_test.out &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --best_epoch=50 --hid_dim=256 --enc_pf_dim=256 --dec_pf_dim=256 --enc_layers=4 --dec_layers=4 --exp_name=nlayers4hdim256 > nlayers4hdim256_test.out &

CUDA_VISIBLE_DEVICES=1 nohup python train.py --best_epoch=45 --hid_dim=256 --enc_pf_dim=256 --dec_pf_dim=256 --enc_layers=3 --dec_layers=3 --enc_heads=10 --dec_heads=10 --exp_name=nlayers3hdim256nhead10 > nlayers3hdim256head10_test.out &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --best_epoch=50 --hid_dim=256 --enc_pf_dim=256 --dec_pf_dim=256 --enc_layers=3 --dec_layers=3 --enc_heads=4 --dec_heads=4 --exp_name=nlayers3hdim256nhead4 > nlayers3hdim256nhead4_test.out &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --best_epoch=50 --hid_dim=256 --enc_pf_dim=256 --dec_pf_dim=256 --enc_layers=3 --dec_layers=3 --enc_heads=6 --dec_heads=6 --exp_name=nlayers3hdim256nhead6 > nlayers3hdim256head6_test.out &
