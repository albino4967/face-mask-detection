echo 'env settings ... '

pip install -r requirements.txt

echo 'download mask dataset ... '
python data_load.py

echo 'unzip dataset ... '
unzip mask_detection_dataset.zip

python mask_detection.py \
  --epoch 30 \
  --weight_decay 0.0005 \
  --learning_rate 0.001 \
  --momentum 0.9

echo "finished ..."