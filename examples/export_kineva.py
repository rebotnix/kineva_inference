from kineva import KINEVA

#initialize model
model = KINEVA(model="models/kineva_coco.pth", classes="./data/coco_classes_kineva.json")

#export model to trt
model.export()
