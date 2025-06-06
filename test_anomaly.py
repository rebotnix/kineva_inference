from kineva import ANOMALY

#initialize model
model = ANOMALY(model="models/metauas_512.trt")

#set reference image
model.set_reference(reference="images/036.png")

#run inference on other image
result_mask, result_score = model.detect("images/024.png")

#print anomaly score
print("Score: "+str(result_score))

#draw heatmap of detection
model.draw(result_mask, "output_anomaly.jpg")
