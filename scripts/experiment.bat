set version=v8
set n_epochs=100
set wandbproj=practical_work_%version%
call conda activate practical_work

echo Run Standard Training
python ./src/train.py --wandb_name standard_d --wandb_project %wandbproj% --model_name standard_d_%version% --n_epochs %n_epochs%

echo Run Robust Training
python ./src/train.py --wandb_name robust_d --include_adversary True --wandb_project %wandbproj% --model_name robust_d_%version% --n_epochs %n_epochs% --adversary_epsilon 2

echo Create Robust Dataset
python ./src/modify_data.py --modification dr --model robust_d_%version%/best.pt --save_path ./data/d_r

echo Create Non-Robust Dataset
python ./src/modify_data.py --modification dnr --model standard_d_%version%/best.pt --save_path ./data/d_nr

echo Create Deterministic Dataset
python ./src/modify_data.py --modification ddet --model standard_d_%version%/best.pt --save_path ./data/d_det

echo Create Deterministic Dataset based on Robust Model
python ./src/modify_data.py --modification ddet --model robust_d_%version%/best.pt --save_path ./data/d_det_robust

echo Create Random Dataset
python ./src/modify_data.py --modification drand --model standard_d_%version%/best.pt --save_path ./data/d_rand

echo Create Random Dataset based on Robust Model
python ./src/modify_data.py --modification drand --model robust_d_%version%/best.pt --save_path ./data/d_rand_robust

echo Standard Train in Dr 
python ./src/train.py --wandb_name standard_dr --wandb_project %wandbproj% --model_name standard_dr_%version% --dataset_path ./data/d_r --n_epochs %n_epochs%

echo Standard Train in Dnr 
python ./src/train.py --wandb_name standard_dnr --wandb_project %wandbproj% --model_name standard_dnr_%version% --dataset_path ./data/d_nr  --n_epochs %n_epochs% 

echo Standard Train in Ddet
python ./src/train.py --wandb_name standard_det_eps5 --wandb_project %wandbproj% --model_name standard_ddet_%version% --dataset_path ./data/d_det --no_augment True --n_epochs %n_epochs% 

echo Standard Train in Ddet_robust
python ./src/train.py --wandb_name standard_det_robust_eps5 --wandb_project %wandbproj% --model_name standard_ddet_robust_%version% --dataset_path ./data/d_det_robust --no_augment True --n_epochs %n_epochs%

echo Standard Train in Drand
python ./src/train.py --wandb_name standard_drand_eps5 --wandb_project %wandbproj% --model_name standard_drand_%version% --dataset_path ./data/d_rand --start_lr 0.01 --n_epochs %n_epochs%

echo Standard Train in Drand_robust
python ./src/train.py --wandb_name standard_drand_robust_eps5 --wandb_project %wandbproj% --model_name standard_drand_robust_%version% --dataset_path ./data/d_rand_robust --start_lr 0.01 --n_epochs %n_epochs%

echo Finished full experiment!
