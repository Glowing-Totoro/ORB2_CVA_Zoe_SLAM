# 
# python test_on_eval.py --batch_size 1  --pose_ext gt pretrained/ablation/abl01_baseline.ckpt

# python test_on_eval.py --batch_size 1  --pose_ext gt pretrained/ablation/abl03_view_aggregation.ckpt
python test_on_eval.py --batch_size 1  --pose_ext gt pretrained/ablation/abl03_view_aggregation.ckpt
# python test_on_eval.py --batch_size 1  --pose_ext gt pretrained/scannet/scannet_tandem.ckpt

# python test_on_eval.py --batch_size 1  --pose_ext gt pretrained/ablation/abl02_vo_window.ckpt




# python test_on_eval.py --batch_size 1 --data_dir /mnt/e/Dataset/tandem_replica_1.1.beta/tandem_replica --pose_ext gt \
  # --tuples_ext dso_optimization_windows_last3 pretrained/ablation/abl01_baseline.ckpt