from typing import List, Tuple

import torch
from PIL import Image

from surya.common.predictor import BasePredictor
from surya.layout.schema import LayoutBox, LayoutResult
from surya.settings import settings
from surya.foundation import FoundationPredictor, TaskNames
from surya.foundation.util import prediction_to_polygon_batch
from surya.input.processing import convert_if_not_rgb
from surya.layout.label import LAYOUT_PRED_RELABEL
from surya.common.util import clean_boxes


LAYOUT_QUANT_BINS = 1024

def skew_to_polygon(
    skew: List[int],
    orig_size: Tuple[int, int],
    new_size: Tuple[int, int],
    min_skew: float = 0.01,
):
    cx, cy, width, height, x_skew, y_skew = skew
    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2
    skew_x = (x_skew - orig_size[0] // 2) / 2
    skew_y = (y_skew - orig_size[1] // 2) / 2

    # Ensures we don't get slightly warped boxes
    # Note that the values are later scaled, so this is in 1/1024 space
    if abs(skew_x) < min_skew:
        skew_x = 0

    if abs(skew_y) < min_skew:
        skew_y = 0

    polygon = [
        x1 - skew_x,
        y1 - skew_y,
        x2 - skew_x,
        y1 + skew_y,
        x2 + skew_x,
        y2 + skew_y,
        x1 + skew_x,
        y2 - skew_y,
    ]

    x_scale = new_size[0] / orig_size[0]
    y_scale = new_size[1] / orig_size[1]

    poly = []
    for i in range(4):
        poly.append([polygon[2 * i] * x_scale, polygon[2 * i + 1] * y_scale])
    return poly

class LayoutPredictor(BasePredictor):
    batch_size = settings.LAYOUT_BATCH_SIZE
    default_batch_sizes = {"cpu": 4, "mps": 4, "cuda": 32, "xla": 16}

    # Override base init - Do not load model
    def __init__(self, foundation_predictor: FoundationPredictor):
        self.foundation_predictor = foundation_predictor
        self.processor = self.foundation_predictor.processor
        self.bbox_size = self.foundation_predictor.model.config.bbox_size
        self.tasks = self.foundation_predictor.tasks

    # Special handling for disable tqdm to pass into foundation predictor
    # Make sure they are kept in sync
    @property
    def disable_tqdm(self) -> bool:
        return super().disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value: bool) -> None:
        self._disable_tqdm = bool(value)
        self.foundation_predictor.disable_tqdm = bool(value)

    def dequantize_bbox_tokens(
        self,
        batch_token_ids: List[torch.Tensor],
        processor,
        image_sizes: List[Tuple[int, int]],
        bbox_size: int,
        bins: int = LAYOUT_QUANT_BINS,
    ):
        """Convert quantized layout tokens back into polygons/bboxes."""

        token_offset = processor.vocab_size
        max_token_value = token_offset + bins
        scale = bbox_size / float(bins - 1)

        predictions = []

        for i, token_ids in enumerate(batch_token_ids):
            image_size = image_sizes[i]
            quant_buffer = []
            batch_predictions = []
            for token in token_ids:
                if token_offset <= token < max_token_value:
                    quant_buffer.append(token - token_offset)
                    continue

                if not quant_buffer:
                    continue

                if len(quant_buffer) == 6:
                    skew = [value * scale for value in quant_buffer]
                    polygon = skew_to_polygon(skew, (bbox_size, bbox_size), image_size)
                    
                    xs = [point[0] for point in polygon]
                    ys = [point[1] for point in polygon]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    label = processor.decode([token], "layout")

                    if label:  # Only keep boxes with a class label
                        batch_predictions.append(
                            {
                                "class_token": token,
                                "class_label": label,
                                "bbox": bbox,
                                "polygon": polygon,
                                "quantized": quant_buffer.copy(),
                            }
                        )

                quant_buffer = []

            predictions.append(batch_predictions)

        return predictions

    def __call__(
        self, images: List[Image.Image], batch_size: int | None = None, top_k: int = 5
    ) -> List[LayoutResult]:
        assert all([isinstance(image, Image.Image) for image in images])
        if batch_size is None:
            batch_size = self.get_batch_size()

        if len(images) == 0:
            return []

        images = convert_if_not_rgb(images)
        images = [self.processor.image_processor(image) for image in images]

        predicted_tokens, batch_bboxes, scores, topk_scores = (
            self.foundation_predictor.prediction_loop(
                images=images,
                input_texts=["" for _ in range(len(images))],
                task_names=[TaskNames.layout for _ in range(len(images))],
                batch_size=batch_size,
                max_lookahead_tokens=0,  # Do not do MTP for layout
                top_k=5,
                max_sliding_window=2148,
                max_tokens=2048,
                tqdm_desc="Recognizing Layout"
            )
        )
        
        image_sizes = [img.shape[:2][::-1] for img in images]
        predicted_polygons = self.dequantize_bbox_tokens(predicted_tokens, self.processor, image_sizes, self.bbox_size, LAYOUT_QUANT_BINS)

        layout_results = []
        for image, image_tokens, polygons, image_scores, image_topk_scores in zip(
            images, predicted_tokens, predicted_polygons, scores, topk_scores
        ):
            layout_boxes = []
            for polygon in polygons:
                label = LAYOUT_PRED_RELABEL.get(polygon["class_label"])
                if not label:
                    continue

                layout_boxes.append(LayoutBox(
                    polygon=polygon["polygon"],
                    label=label,
                    position=0,
                    top_k={label: 1},
                    confidence=1,
                ))
            layout_boxes = clean_boxes(layout_boxes)
            layout_results.append(LayoutResult(bboxes=layout_boxes, image_bbox=[0, 0, image.shape[1], image.shape[0]]))


        assert len(layout_results) == len(images)
        return layout_results
