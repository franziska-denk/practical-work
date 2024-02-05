set version=v15
set seed=47
set n_epochs=150
set wandbproj=practical_work_%version%
rem set eps=0.0313
rem set alpha=0.0078
set eps=0.5
set alpha=0.1
call conda activate practical_work

echo Run Standard Training
python ./src/train.py --wandb_name standard_d --wandb_project %wandbproj% --model_name standard_d_%version% --n_epochs %n_epochs% --adversary_epsilon %eps% --seed %seed%

echo Run Robust Training
python ./src/train.py --wandb_name robust_d_eps%eps% --include_adversary True --wandb_project %wandbproj% --model_name robust_d_%version% --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Create Robust Dataset
python ./src/modify_data.py --modification dr --model robust_d_%version%/best_adv.pt --save_path ./data/d_r_%version% --seed %seed%

echo Create Non-Robust Dataset
python ./src/modify_data.py --modification dnr --model standard_d_%version%/best.pt --save_path ./data/d_nr --seed 11

echo Create Deterministic Dataset
python ./src/modify_data.py --modification ddet --model standard_d_%version%/best.pt --save_path ./data/d_det_eps%eps% --epsilon %eps% --alpha %alpha% --seed %seed%

echo Create Deterministic Dataset based on Robust Model
python ./src/modify_data.py --modification ddet --model robust_d_%version%/best.pt --save_path ./data/d_det_robust_eps%eps%  --epsilon %eps% --alpha %alpha% --seed %seed%

echo Create Random Dataset
python ./src/modify_data.py --modification drand --model standard_d_%version%/best.pt --save_path ./data/d_rand_eps%eps% --epsilon %eps% --alpha %alpha% --seed %seed%

echo Create Random Dataset based on Robust Model
python ./src/modify_data.py --modification drand --model robust_d_%version%/best.pt --save_path ./data/d_rand_robust_eps%eps% --epsilon %eps% --alpha %alpha% --seed %seed%

echo Standard Train in Dr 
python ./src/train.py --wandb_name standard_dr --wandb_project %wandbproj% --model_name standard_dr_%version% --dataset_path ./data/d_r --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Standard Train in Dnr 
python ./src/train.py --wandb_name standard_dnr_eps%eps% --wandb_project %wandbproj% --model_name standard_dnr_%version% --dataset_path ./data/d_nr  --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Standard Train in Ddet
python ./src/train.py --wandb_name standard_det_eps%eps% --wandb_project %wandbproj% --model_name standard_ddet_eps%eps%_%version% --dataset_path ./data/d_det_eps%eps% --no_augment True --n_epochs %n_epochs%  --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Standard Train in Ddet_robust
python ./src/train.py --wandb_name standard_det_robust_eps%eps% --wandb_project %wandbproj% --model_name standard_ddet_robust_eps%eps%_%version% --dataset_path ./data/d_det_robust_eps%eps% --no_augment True --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Standard Train in Drand
python ./src/train.py --wandb_name standard_drand_eps%eps% --wandb_project %wandbproj% --model_name standard_drand_eps%eps%_%version% --dataset_path ./data/d_rand_eps%eps% --start_lr 0.01 --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Standard Train in Drand_robust
python ./src/train.py --wandb_name standard_drand_robust_eps%eps% --wandb_project %wandbproj% --model_name standard_drand_eps%eps%_robust_%version% --dataset_path ./data/d_rand_robust_eps%eps% --start_lr 0.01 --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

set eps=1
echo Run Robust Training
python ./src/train.py --wandb_name robust_d_eps%eps% --include_adversary True --wandb_project %wandbproj% --model_name robust_d_%version%_eps%eps% --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Create Deterministic Dataset based on Robust Model
python ./src/modify_data.py --modification ddet --model robust_d_%version%_eps%eps%/best.pt --save_path ./data/d_det_robust_eps%eps%  --epsilon %eps% --alpha %alpha% --seed %seed%

echo Create Random Dataset based on Robust Model
python ./src/modify_data.py --modification drand --model robust_d_%version%_eps%eps%/best.pt --save_path ./data/d_rand_robust_eps%eps% --epsilon %eps% --alpha %alpha% --seed %seed%

echo Standard Train in Drand_robust
python ./src/train.py --wandb_name standard_drand_robust_eps%eps% --wandb_project %wandbproj% --model_name standard_drand_eps%eps%_robust_%version% --dataset_path ./data/d_rand_robust_eps%eps% --start_lr 0.01 --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Standard Train in Ddet_robust
python ./src/train.py --wandb_name standard_det_robust_eps%eps% --wandb_project %wandbproj% --model_name standard_ddet_robust_eps%eps%_%version% --dataset_path ./data/d_det_robust_eps%eps% --no_augment True --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Standard Train in Ddet_robust
python ./src/train.py --wandb_name standard_det_robust_eps%eps% --wandb_project %wandbproj% --model_name standard_ddet_robust_eps%eps%_%version% --dataset_path ./data/d_det_robust_eps%eps% --no_augment True --n_epochs %n_epochs% --adversary_epsilon %eps% --adversary_alpha %alpha% --seed %seed%

echo Create Deterministic Dataset
python ./src/modify_data.py --modification ddet --model standard_d_%version%/best.pt --save_path ./data/d_det_eps%eps% --epsilon %eps% --seed %seed%

echo Create Random Dataset  
python ./src/modify_data.py --modification drand --model standard_d_%version%/best.pt --save_path ./data/d_rand_eps_%eps% --epsilon %eps% --seed %seed%

echo Standard Train in Ddet
python ./src/train.py --wandb_name standard_det_eps%eps% --wandb_project %wandbproj% --model_name standard_ddet_eps%eps%_%version% --dataset_path ./data/d_det_eps%eps% --no_augment True --n_epochs %n_epochs% --seed %seed%

echo Standard Train in Drand
python ./src/train.py --wandb_name standard_drand_eps%eps% --wandb_project %wandbproj% --model_name standard_drand_eps%eps%_%version% --dataset_path ./data/d_rand_eps_%eps% --start_lr 0.01 --n_epochs %n_epochs% --seed %seed%

set eps=2
echo Create Deterministic Dataset
python ./src/modify_data.py --modification ddet --model standard_d_%version%/best.pt --save_path ./data/d_det_eps%eps% --epsilon %eps% --seed %seed%

echo Create Random Dataset  
python ./src/modify_data.py --modification drand --model standard_d_%version%/best.pt --save_path ./data/d_rand_eps_%eps% --epsilon %eps% --seed %seed%

echo Standard Train in Ddet
python ./src/train.py --wandb_name standard_det_eps%eps% --wandb_project %wandbproj% --model_name standard_ddet_eps%eps%_%version% --dataset_path ./data/d_det_eps%eps% --no_augment True --n_epochs %n_epochs% --seed %seed%

echo Standard Train in Drand
python ./src/train.py --wandb_name standard_drand_eps%eps% --wandb_project %wandbproj% --model_name standard_drand_eps%eps%_%version% --dataset_path ./data/d_rand_eps_%eps% --start_lr 0.01 --n_epochs %n_epochs% --seed %seed%

set eps=4
echo Create Deterministic Dataset
python ./src/modify_data.py --modification ddet --model standard_d_%version%/best.pt --save_path ./data/d_det_eps%eps% --epsilon %eps% --seed %seed%

echo Create Random Dataset
python ./src/modify_data.py --modification drand --model standard_d_%version%/best.pt --save_path ./data/d_rand_eps_%eps% --epsilon %eps% --seed %seed%

echo Standard Train in Ddet
python ./src/train.py --wandb_name standard_det_eps%eps% --wandb_project %wandbproj% --model_name standard_ddet_eps%eps%_%version% --dataset_path ./data/d_det_eps%eps% --no_augment True --n_epochs %n_epochs% --seed %seed%

echo Standard Train in Drand
python ./src/train.py --wandb_name standard_drand_eps%eps% --wandb_project %wandbproj% --model_name standard_drand_eps%eps%_%version% --dataset_path ./data/d_rand_eps_%eps%  --start_lr 0.01 --n_epochs %n_epochs% --seed %seed%
echo Finished full experiment!
