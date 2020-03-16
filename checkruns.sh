#python Standard_Training.py --savename resnet_cub_margin_semihard     --source_path /home/karsten_dl/Dropbox/Projects/Datasets --dataset cub200 --n_epochs 60 --tau 60 --loss marginloss --sampling semihard --gpu 1
#python Standard_Training.py --savename resnet_cub_margin_hardsampling --source_path /home/karsten_dl/Dropbox/Projects/Datasets --dataset cub200 --n_epochs 60 --tau 60 --loss marginloss --sampling softhard --gpu 1
#python Standard_Training.py --savename resnet_car_margin_semihard     --source_path /home/karsten_dl/Dropbox/Projects/Datasets --dataset cars196 --n_epochs 70 --tau 70 --loss marginloss --sampling semihard --gpu 1
#python Standard_Training.py --savename resnet_car_margin_hardsampling --source_path /home/karsten_dl/Dropbox/Projects/Datasets --dataset cars196 --n_epochs 70 --tau 70 --loss marginloss --sampling softhard --gpu 1


python Standard_Training.py --savename resnet_cub_margin_semihard     --margin 0.2 --source_path /home/karsten_dl/Dropbox/Projects/Datasets --dataset cub200 --n_epochs 60 --tau 60 --loss marginloss --sampling semihard --gpu 1
python Standard_Training.py --savename resnet_car_margin_semihard     --margin 0.2 --source_path /home/karsten_dl/Dropbox/Projects/Datasets --dataset cars196 --n_epochs 70 --tau 70 --loss marginloss --sampling semihard --gpu 1
