from kineva import KINEVA

#initialize model
model = KINEVA(model="models/rb_coco.pth")

#run inference on image
final_boxes, final_scores, final_labels = model.detect("images/bus.jpg", threshold=0.35)

#draw detection
model.draw(final_boxes, final_scores, final_labels, output_path="./outputs/output_kineva.jpg")
