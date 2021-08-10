# python3 train_PartialSSIM_.py --cuda --gpus 1 --code L 
# python3 train_PartialSSIM_.py --cuda --gpus 1 --code C 
# python3 train_PartialSSIM_.py --cuda --gpus 1 --code S 
# python3 train_PartialSSIM_.py --cuda --gpus 1 --code CS
# python3 train_PartialSSIM_.py --cuda --gpus 1 --code LC
# python3 train_PartialSSIM_.py --cuda --gpus 1 --code LS

python3 inference_.py --gpus 1 --loss_type ssim_loss_L
python3 inference_.py --gpus 1 --loss_type ssim_loss_C
python3 inference_.py --gpus 1 --loss_type ssim_loss_S
python3 inference_.py --gpus 1 --loss_type ssim_loss_LC
python3 inference_.py --gpus 1 --loss_type ssim_loss_LS
python3 inference_.py --gpus 1 --loss_type ssim_loss_CS

python3 inference_.py --gpus 1 --loss_type ssim_loss_L_
python3 inference_.py --gpus 1 --loss_type ssim_loss_C_
python3 inference_.py --gpus 1 --loss_type ssim_loss_S_
python3 inference_.py --gpus 1 --loss_type ssim_loss_LC_
python3 inference_.py --gpus 1 --loss_type ssim_loss_LS_
python3 inference_.py --gpus 1 --loss_type ssim_loss_CS_
