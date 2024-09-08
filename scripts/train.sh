ACC_CONFIG="./configs/accelerate-train.yaml"
TRAIN_CONFIG="./configs/train-panohead-large.yaml"

accelerate launch --config_file $ACC_CONFIG -m openlrm.launch train.lrm --config $TRAIN_CONFIG