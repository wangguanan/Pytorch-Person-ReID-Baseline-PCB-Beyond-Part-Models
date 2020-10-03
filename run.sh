rlaunch -P 0 --charged-group=face_det_terminal --cpu 16 --gpu 2 --memory 20480 -- python main.py \
 --market_path /data/datasets/Market-1501-v15.09.15 --duke_path /data/datasets/DukeMTMC-reID \
 --train_dataset market --test_dataset market \
 --output_path ./results/market/

 rlaunch -P 0 --charged-group=face_det_terminal --cpu 16 --gpu 2 --memory 20480 -- python main.py \
 --market_path /data/datasets/Market-1501-v15.09.15 --duke_path /data/datasets/DukeMTMC-reID \
 --train_dataset duke --test_dataset duke \
  --output_path ./results/duke/
