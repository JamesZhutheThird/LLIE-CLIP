## CLIP: Cascaded Learning with Inception Pattern on Low-Light Image Enhancement

A group project of CS7303 (2022-2023 autumn)


### requirements

```
python3.10
pytorch==1.11.0
wandb
```
### how to run

0. Download our combined datasets from [Baiduyun](https:// "title"), unzip them and place them under directory ```Datasets```. You can also use your own datasets and modify the training script. The dataset we use are selected from [DarkFace](http://cvpr2022.ug2challenge.org/program21/dataset21_t1.html), [GLADNet](https://daooshee.github.io/fgworkshop18Gladnet/) and [LOL Dataset](https://flyywh.github.io/IJCV2021LowLight_VELOL/).

1. To run traditional algorithms
```shell
python test_classic.py --data DATA --mode MODE 
```


2. To run CLIP model with slurm and DDP, simply use 
```shell
sbatch run_ddp_new.sh 
```
 You can run this script with shell locally, by setting ```SLURM_ARRAY_TASK_ID``` to any preset index. 
```shell
sh run_ddp_new.sh 
```
You could also change one or multiple arguments in the script, and you can find their descriptions in ```train_ddp.py```. 
