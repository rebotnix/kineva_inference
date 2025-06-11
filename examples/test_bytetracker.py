from kineva import BYTE_TRACKER, KINEVA
#import supervision as sv

#initialize model
model = KINEVA(model="models/kineva_coco.trt", classes="./data/coco_classes_rfdetr.json")

#initialize tracker
tracker = BYTE_TRACKER()
#tracker_o = sv.ByteTrack()

#run inference on image
final_boxes, final_scores, final_labels = model.detect("images/bus.jpg", threshold=0.35)

trackind_ids = None

for el in range(0,60):
  trackind_ids = tracker.update_with_detections(final_boxes, final_scores, final_labels, trackind_ids)
  print(trackind_ids)

