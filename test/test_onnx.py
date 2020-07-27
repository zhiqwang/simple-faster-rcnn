import io
import torch
import ops

from models.image_list import ImageList
from models.transform import GeneralizedRCNNTransform
from models.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from models.roi_heads import RoIHeads
from models.faster_rcnn import FastRCNNPredictor, TwoMLPHead

from collections import OrderedDict

# onnxruntime requires python 3.5 or above
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

import unittest
from ops._register_onnx_ops import _onnx_opset_version


@unittest.skipIf(onnxruntime is None, 'ONNX Runtime unavailable')
class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(self, model, inputs_list, tolerate_small_mismatch=False, do_constant_folding=True, dynamic_axes=None,
                  output_names=None, input_names=None):
        model.eval()

        onnx_io = io.BytesIO()
        # export to onnx with the first input
        torch.onnx.export(model, inputs_list[0], onnx_io,
                          do_constant_folding=do_constant_folding, opset_version=_onnx_opset_version,
                          dynamic_axes=dynamic_axes, input_names=input_names, output_names=output_names)
        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, torch.Tensor) or \
                   isinstance(test_inputs, list):
                    test_inputs = (test_inputs,)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)
            self.ort_validate(onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch)

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # compute onnxruntime output prediction
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)
        for i in range(0, len(outputs)):
            try:
                torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

    @unittest.skip("Disable test until Split w/ zero sizes is implemented in ORT")
    def test_new_empty_tensor(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()
                self.conv2 = ops.misc.ConvTranspose2d(16, 33, (3, 5))

            def forward(self, input2):
                return self.conv2(input2)

        input = torch.rand(0, 16, 10, 10)
        test_input = torch.rand(0, 16, 20, 20)
        self.run_model(Module(), [(input, ), (test_input,)], do_constant_folding=False)

    def test_clip_boxes_to_image(self):
        boxes = torch.randn(5, 4) * 500
        boxes[:, 2:] += boxes[:, :2]
        size = torch.randn(200, 300)

        size_2 = torch.randn(300, 400)

        class Module(torch.nn.Module):
            def forward(self, boxes, size):
                return ops.boxes.clip_boxes_to_image(boxes, size.shape)

        self.run_model(Module(), [(boxes, size), (boxes, size_2)],
                       input_names=["boxes", "size"],
                       dynamic_axes={"size": [0, 1]})

    def _init_test_generalized_rcnn_transform(self):
        min_size = 100
        max_size = 200
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        return transform

    def _init_test_rpn(self):
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        out_channels = 256
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=1000)
        rpn_post_nms_top_n = dict(training=2000, testing=1000)
        rpn_nms_thresh = 0.7

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        return rpn

    def _init_test_roi_heads_faster_rcnn(self):
        out_channels = 256
        num_classes = 91

        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None
        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100

        box_roi_pool = ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        return roi_heads

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [
            ('0', torch.rand(2, 256, s0 // 4, s1 // 4)),
            ('1', torch.rand(2, 256, s0 // 8, s1 // 8)),
            ('2', torch.rand(2, 256, s0 // 16, s1 // 16)),
            ('3', torch.rand(2, 256, s0 // 32, s1 // 32)),
            ('4', torch.rand(2, 256, s0 // 64, s1 // 64)),
        ]
        features = OrderedDict(features)
        return features

    def test_rpn(self):
        class RPNModule(torch.nn.Module):
            def __init__(self_module):
                super(RPNModule, self_module).__init__()
                self_module.rpn = self._init_test_rpn()

            def forward(self_module, images, features):
                images = ImageList(images, [i.shape[-2:] for i in images])
                return self_module.rpn(images, features)

        images = torch.rand(2, 3, 150, 150)
        features = self.get_features(images)
        images2 = torch.rand(2, 3, 80, 80)
        test_features = self.get_features(images2)

        model = RPNModule()
        model.eval()
        model(images, features)

        self.run_model(model, [(images, features), (images2, test_features)], tolerate_small_mismatch=True,
                       input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
                       dynamic_axes={"input1": [0, 1, 2, 3], "input2": [0, 1, 2, 3],
                                     "input3": [0, 1, 2, 3], "input4": [0, 1, 2, 3],
                                     "input5": [0, 1, 2, 3], "input6": [0, 1, 2, 3]})

    def test_roi_heads(self):
        class RoiHeadsModule(torch.nn.Module):
            def __init__(self_module):
                super(RoiHeadsModule, self_module).__init__()
                self_module.transform = self._init_test_generalized_rcnn_transform()
                self_module.rpn = self._init_test_rpn()
                self_module.roi_heads = self._init_test_roi_heads_faster_rcnn()

            def forward(self_module, images, features):
                original_image_sizes = [img.shape[-2:] for img in images]
                images = ImageList(images, [i.shape[-2:] for i in images])
                proposals, _ = self_module.rpn(images, features)
                detections, _ = self_module.roi_heads(features, proposals, images.image_sizes)
                detections = self_module.transform.postprocess(detections,
                                                               images.image_sizes,
                                                               original_image_sizes)
                return detections

        images = torch.rand(2, 3, 100, 100)
        features = self.get_features(images)
        images2 = torch.rand(2, 3, 150, 150)
        test_features = self.get_features(images2)

        model = RoiHeadsModule()
        model.eval()
        model(images, features)

        self.run_model(model, [(images, features), (images2, test_features)], tolerate_small_mismatch=True,
                       input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
                       dynamic_axes={"input1": [0, 1, 2, 3], "input2": [0, 1, 2, 3], "input3": [0, 1, 2, 3],
                                     "input4": [0, 1, 2, 3], "input5": [0, 1, 2, 3], "input6": [0, 1, 2, 3]})
