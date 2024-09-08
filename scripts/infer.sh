# Example usage
EXPORT_VIDEO=true
EXPORT_MESH=true
INFER_CONFIG="./configs/infer-b.yaml"
#MODEL_NAME="zxhezexin/openlrm-mix-large-1.1"
MODEL_NAME="expr/releases/lrm-objaverse/small-dummyrun/step_004680"
IMAGE_INPUT="/home/jianjinx/data2/HHRGaussian-priv/data/PanoHeadDense/064061/images/IMG_0000.jpg"

python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH