ACC_CONFIG="./configs/accelerate-train.yaml"
TRAIN_CONFIG="./configs/train-panohead-large.yaml"
LOG_LEVEL="INFO"

accelerate launch --config_file $ACC_CONFIG -m openlrm.launch train.lrm --config $TRAIN_CONFIG