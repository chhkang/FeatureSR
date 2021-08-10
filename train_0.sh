# python3 train_PartialSSIM.py --cuda --gpus 0 --code L 
# python3 train_PartialSSIM.py --cuda --gpus 0 --code C
# python3 train_PartialSSIM.py --cuda --gpus 0 --code S 
# python3 train_PartialSSIM.py --cuda --gpus 0 --code CS 
# python3 train_PartialSSIM.py --cuda --gpus 0 --code LC 
# python3 train_PartialSSIM.py --cuda --gpus 0 --code LS 

python3 inference.py --gpus 0 --loss_type ssim_loss_L
python3 inference.py --gpus 0 --loss_type ssim_loss_C
python3 inference.py --gpus 0 --loss_type ssim_loss_S
python3 inference.py --gpus 0 --loss_type ssim_loss_LC
python3 inference.py --gpus 0 --loss_type ssim_loss_LS
python3 inference.py --gpus 0 --loss_type ssim_loss_CS

python3 inference.py --gpus 0 --loss_type ssim_loss_L_
python3 inference.py --gpus 0 --loss_type ssim_loss_C_
python3 inference.py --gpus 0 --loss_type ssim_loss_S_
python3 inference.py --gpus 0 --loss_type ssim_loss_LC_
python3 inference.py --gpus 0 --loss_type ssim_loss_LS_
python3 inference.py --gpus 0 --loss_type ssim_loss_CS_
