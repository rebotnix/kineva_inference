from kineva import ANOMALY

#initialize model
model = ANOMALY(model="models/metauas_512.pth")

#export model to trt
model.export()
