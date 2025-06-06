from kineva import KINEVA
import time

#initialize model with reference image
model = KINEVA(model="models/kineva_coco.trt")

for i in range(0,10):
  st = time.time()
  final_boxes, final_scores, final_labels = model.detect("images/bus.jpg", threshold=0.35)
  print(time.time()-st)
#draw
model.draw(final_boxes, final_scores, final_labels, output_path="output_kineva.jpg")
