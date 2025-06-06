from kineva import KINEVA

#initialize model
model = KINEVA(model="models/kineva_coco.pth")

#export model
model.export()
