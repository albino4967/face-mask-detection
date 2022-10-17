echo 'env settings ... '

apt install virtualenv

VENV_PATH="${SCRIPT_DIR}/venv"
if [ ! -d ${VENV_PATH} ]; then
  virtualenv -p python3.6 "${VENV_PATH}"
  source ${VENV_PATH}/bin/activate
  pip install -r requirements.txt
else
  echo "${VENV_PATH} already exists. Skipping virtual environment setup."
  source ${VENV_PATH}/bin/activate
fi


echo 'download mask dataset ... '
python data_load.py

echo 'unzip dataset ... '
nuzip mask_detection_dataset.zip

python mask_detection.py \
  --epoch 10 \
  --weight_decay 0.0005 \
  --learning_rate 0.001 \
  --momentum 0.9

echo "finished ..."