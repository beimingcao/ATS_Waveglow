#!/bin/bash -e

new_experiment=False
training=False
testing=True

conf_dir=conf/ATS_conf.yaml
experiments_dir=experiments
current_exp=current_exp

if [ "$new_experiment" = "True" ];then
echo "New experiments, loading data into current_exp folder"
sudo rm -rf $current_exp
python3 0_setup.py --conf_dir $conf_dir --exp_dir $experiments_dir --buff_dir $current_exp
python3 1_data_prepare.py --conf_dir $conf_dir --buff_dir $current_exp
python3 2_data_loadin.py --conf_dir $conf_dir --buff_dir $current_exp
fi
if [ "$training" = "True" ];then
python3 3_train.py --conf_dir $conf_dir --buff_dir $current_exp
fi

if [ "$testing" = "True" ];then
python3 4_test.py --conf_dir $conf_dir --buff_dir $current_exp
fi
