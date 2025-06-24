from kineva import RFDETR

#initialize model
model = RFDETR(model="models/rb_coco.trt", classes="./data/coco_classes_rfdetr.json")

#run inference on image
final_boxes, final_scores, final_labels = model.detect("images/bus.jpg", threshold=0.5)

#draw detection
model.draw(final_boxes, final_scores, final_labels, output_path="./outputs/output_rfdetr.jpg")
