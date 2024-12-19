"""
Pytoch native backend 
Extended by Grigori Fursin for the ABTF demo
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import torchvision
import backend

import os
import sys
import importlib

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend")

class BackendPytorchNative(backend.Backend):
    def __init__(self):
        super(BackendPytorchNative, self).__init__()
        self.sess = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Grigori added for ABTF model
        self.config = None
        self.num_classes = None
        self.image_size = None


    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):

        # From ABTF code
        sys.path.insert(0, os.environ['CM_ML_MODEL_CODE_WITH_PATH'])

        from src.transform import SSDTransformer
        from src.utils import generate_dboxes, Encoder, colors, coco_classes
        from src.model import SSD, ResNet

        abtf_model_config = os.environ.get('CM_ABTF_ML_MODEL_CONFIG', '')

        num_classes_str = os.environ.get('CM_ABTF_NUM_CLASSES', '').strip()
        self.num_classes = int(num_classes_str) if num_classes_str!='' else 15

        self.config = importlib.import_module('config.' + abtf_model_config)
        self.image_size = self.config.model['image_size']

        # self.model = SSD(self.config.model, backbone=ResNet(self.config.model), num_classes=self.num_classes)

        # Modify (Xiangyu): replace torch.load with openvino read_model()
        # checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        # self.model.load_state_dict(checkpoint["model_state_dict"])

        # if self.device.startswith('cuda'):
        #     self.model.cuda()

        # self.model.eval()
        # print(self.model)

        # self.model = self.model.to(self.device)
        import openvino as ov
        core = ov.Core()
        self.model = core.compile_model(model_path, "AUTO")
        self.input_layer_ir = self.model.input(0)
        self.output_layer_ir_0 = self.model.output(0)
        self.output_layer_ir_1 = self.model.output(1)

        self.inputs = inputs
        self.outputs = outputs


        return self


    def predict(self, feed):
        # For ABTF

        # Note(Xiangyu): Haven't supported batching yet. max_batchsize > 1 will crash.
        # Always first element for now (later may stack for batching)
        img = feed['image'][0]

        if torch.cuda.is_available():
            img = img.cuda()

        inp = img.unsqueeze(dim=0)

        with torch.no_grad():
            # Modify (Xiangyu)
            # ploc, plabel = self.model(inp)
            ploc = self.model([inp])[self.output_layer_ir_0]
            plabel = self.model([inp])[self.output_layer_ir_1]
            ploc = torch.from_numpy(ploc)
            plabel = torch.from_numpy(plabel)

            output = (ploc, plabel)

        return output
