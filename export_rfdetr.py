from kineva import RFDETR

#initialize model
model = RFDETR(model="models/rf-detr-base.pth")

#export model
model.export()
