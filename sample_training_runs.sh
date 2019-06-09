################################################################################################################
################################################ ResNet RUNS ###################################################
### ResNet50 on CUB200-2011: MarginLoss, TripletLoss + Semihard, ProxyNCA, N-Pair
python Standard_Training.py --gpu 0 --savename resnet_cub_margin_dist    --dataset cub200 --n_epochs 50 --tau 30 --loss marginloss --sampling distance
python Standard_Training.py --gpu 0 --savename resnet_cub_triplet_semi   --dataset cub200 --n_epochs 50 --tau 30 --loss triplet    --sampling semihard
python Standard_Training.py --gpu 0 --savename resnet_cub_proxynca_none  --dataset cub200 --n_epochs 40 --tau 25 --loss proxynca
python Standard_Training.py --gpu 0 --savename resnet_cub_npair_none     --dataset cub200 --n_epochs 55 --tau 35 --loss npair

### ResNet50 on CARS196: MarginLoss, TripletLoss + Semihard, ProxyNCA, N-Pair
python Standard_Training.py --gpu 0 --savename resnet_cars_margin_dist   --dataset cars196 --n_epochs 70 --tau 45 --loss marginloss --sampling distance
python Standard_Training.py --gpu 0 --savename resnet_cars_triplet_semi  --dataset cars196 --n_epochs 70 --tau 45 --loss triplet    --sampling semihard
python Standard_Training.py --gpu 0 --savename resnet_cars_proxynca_none --dataset cars196 --n_epochs 40 --tau 25 --loss proxynca
python Standard_Training.py --gpu 0 --savename resnet_cars_npair_none    --dataset cars196 --n_epochs 85 --tau 55 --loss npair

### ResNet50 on Stanford Online-Products: MarginLoss, TripletLoss + Semihard, N-Pair
python Standard_Training.py --gpu 0 --savename resnet_op_margin_dist     --dataset online_products --n_epochs 40 --tau 30    --loss marginloss --sampling distance
python Standard_Training.py --gpu 0 --savename resnet_op_triplet_semi    --dataset online_products --n_epochs 40 --tau 30    --loss triplet    --sampling semihard
python Standard_Training.py --gpu 0 --savename resnet_op_npair_none      --dataset online_products --n_epochs 25 --tau 15 20 --loss npair

### ResNet50 on In-Shop Clothes: MarginLoss, TripletLoss + Semihard, N-Pair
python Standard_Training.py --gpu 0 --savename resnet_inshop_margin_dist  --dataset in-shop --n_epochs 40 --tau 25 --loss marginloss --sampling distance
python Standard_Training.py --gpu 0 --savename resnet_inshop_triplet_semi --dataset in-shop --n_epochs 40 --tau 25 --loss triplet    --sampling semihard
python Standard_Training.py --gpu 0 --savename resnet_inshop_npair_none   --dataset in-shop --n_epochs 40 --tau 25 --loss npair

# ### (optional) ResNet50 on Vehicle-ID
# python Standard_Training.py --gpu 0 --savename resnet_vehicle_margin_dist   --dataset vehicle_id --n_epochs 40 --tau 25 --loss marginloss --sampling distance
# python Standard_Training.py --gpu 0 --savename resnet_vehicle_triplet_semi  --dataset vehicle_id --n_epochs 40 --tau 25 --loss triplet    --sampling semihard
# python Standard_Training.py --gpu 0 --savename resnet_vehicle_npair_none    --dataset vehicle_id --n_epochs 40 --tau 25 --loss npair






################################################################################################################
############################################# GoogLeNet RUNS ###################################################
### GoogLeNet on CUB200-2011: MarginLoss, TripletLoss + Semihard, ProxyNCA, N-Pair
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_cars_margin_dist   --dataset cars196 --n_epochs 70 --tau 30 50 --loss marginloss --sampling distance
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_cars_triplet_semi  --dataset cars196 --n_epochs 70 --tau 30 50 --loss triplet    --sampling semihard
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_cars_proxynca_none --dataset cars196 --n_epochs 40 --tau 20 30 --loss proxynca
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_cars_npair_none    --dataset cars196 --n_epochs 85 --tau 40 65 --loss npair

### GoogLeNet on CARS196: MarginLoss, TripletLoss + Semihard, ProxyNCA, N-Pair
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_cub_margin_dist   --dataset cub200 --n_epochs 60  --tau 10 20 30 40 50 --loss marginloss --sampling distance
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_cub_triplet_semi  --dataset cub200 --n_epochs 70  --tau 15 40 60         --loss triplet    --sampling semihard
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_cub_proxynca_none --dataset cub200 --n_epochs 25  --tau 7 17              --loss proxynca
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_cub_npair_none    --dataset cub200 --n_epochs 150 --tau 100                --loss npair

### GoogLeNet on Online Products: MarginLoss, TripletLoss + Semihard, N-Pair
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_op_margin_dist   --dataset online_products --n_epochs 20 --tau 2 6 --loss marginloss --sampling distance
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_op_triplet_semi  --dataset online_products --n_epochs 40 --tau 10  --loss triplet    --sampling semihard
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_op_npair_none    --dataset online_products --n_epochs 20 --tau 2 6 --loss npair

### GoogLeNet on In-Shop Clothes: MarginLoss, TripletLoss + Semihard, N-Pair
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_inshop_margin_dist   --dataset in-shop --n_epochs 40 --tau 25 --loss marginloss --sampling distance
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_inshop_triplet_semi  --dataset in-shop --n_epochs 40 --tau 25 --loss triplet    --sampling semihard
python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_inshop_npair_none    --dataset in-shop --n_epochs 40 --tau 25 --loss npair

# ### (optional) GoogLeNet on Vehicle-ID
# python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_vehicle_margin_dist   --dataset vehicle_id --n_epochs 40 --tau 25 --loss marginloss --sampling distance
# python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_vehicle_triplet_semi  --dataset vehicle_id --n_epochs 40 --tau 25 --loss triplet    --sampling semihard
# python Standard_Training.py --gpu 0 --arch googlenet --lr 0.0001 --embed_dim 512 --savename gnet_vehicle_npair_none    --dataset vehicle_id --n_epochs 40 --tau 25 --loss npair






########################################################################################
#################################### Special RUNS ######################################
python Standard_Training.py --gpu 0 --savename resnet_cars_margin_dist   --dataset cars196 --n_epochs 70 --tau 45 --loss marginloss --sampling distance --distance_measure
python Standard_Training.py --gpu 0 --savename resnet_cars_margin_dist   --dataset cars196 --n_epochs 70 --tau 45 --loss marginloss --sampling distance --grad_measure
