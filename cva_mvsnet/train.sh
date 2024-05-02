rm -rf trained_model
mkdir trained_model
export TANDEM_DATA_DIR=/mnt/e/Dataset/tandem_replica_1.1.beta/tandem_replica
python train.py --config configs/default.yaml /home/yu/ORB2_CVA_MVSnet_SLAM_Test/cva_mvsnet/trained_model DATA.ROOT_DIR $TANDEM_DATA_DIR