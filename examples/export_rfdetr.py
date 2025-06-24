from kineva import RFDETR

#initialize model
model = RFDETR(model="models/rb_coco.pth")

#export model to trt
model.export()
