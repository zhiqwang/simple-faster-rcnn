import numpy as np
import cv2
import torch


def select_top_predictions(predictions, threshold):
    idx = (predictions['scores'] > threshold).nonzero().squeeze(1)
    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]
    return new_predictions


def compute_colors_for_labels(labels, palette=None):
    '''
    Simple function that adds fixed colors depending on the class
    '''
    if palette is None:
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype('uint8')
    return colors


def overlay_boxes(image, predictions):
    '''
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    '''
    labels = predictions['labels']
    boxes = predictions['boxes']

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1,
        )

    return image


def overlay_class_names(image, predictions, categories):
    '''
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    '''
    scores = predictions['scores'].tolist()
    labels = predictions['labels'].tolist()
    labels = [categories[i] for i in labels]
    boxes = predictions['boxes']

    template = '{}: {:.2f}'
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
        )

    return image


def predict(img, model, categories, device):
    model = model.eval()

    result = img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 416 x 416
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img)

    with torch.no_grad():
        output = model([img.to(device)])
    top_predictions = select_top_predictions(output[0], 0.7)
    top_predictions = {k: v.cpu() for k, v in top_predictions.items()}

    result = overlay_boxes(result, top_predictions)

    result = overlay_class_names(result, top_predictions, categories)
    return result, output, top_predictions
