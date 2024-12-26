"""
@author: Viet Nguyen <nhviet1009@gmail.com>

Extended by Radoyeh Shojaei and Grigori Fursin
"""

print ('Initializing packages for ABTF PyTorch model...')
print ('')

from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from torchvision import transforms as T
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import openvino as ov

# From ABTF code
from datasets import VOCSegmentation, Cityscapes, cityscapes, Cognata, Waymo
from metrics import StreamSegMetrics

def get_args():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'cognata', 'waymo'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    args = parser.parse_args()
    return args



def test(opt):
    import os

    # # Check remote debugging via CM (tested with Visual Studio)
    # if os.environ.get('CM_TMP_DEBUG_UID', '') == '7cf735bf80204efb':
    #     import cmind.utils
    #     cmind.utils.debug_here(__file__, port=5678).breakpoint()

    if opt.dataset.lower() == 'voc':
        opt.num_classes = 21
        # opt.num_classes = 0
        decode_fn = VOCSegmentation.decode_target
    elif opt.dataset.lower() == 'cityscapes':
        opt.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opt.dataset.lower() == 'cognata':
        opt.num_classes = 19
        decode_fn = Cognata.decode_target
    elif opt.dataset.lower() == 'waymo':
        opt.num_classes = 19
        decode_fn = Waymo.decode_target

    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opt.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opt.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opt.input):
        image_files.append(opt.input)
    
    # # Set up model (all models are 'constructed at network.modeling)
    # model = network.modeling.__dict__[opt.model](num_classes=opt.num_classes, output_stride=opt.output_stride)
    # if opt.separable_conv and 'plus' in opt.model:
    #     network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    import openvino as ov
    core = ov.Core()
    
    if opt.ckpt is not None and os.path.isfile(opt.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        # checkpoint = torch.load(opt.ckpt, map_location=torch.device('cpu'))
        # model.load_state_dict(checkpoint["model_state"])
        # model = nn.DataParallel(model)
        # model.to(device)
        model = core.read_model(opt.ckpt)
        model.reshape([1, 3, 513, 513])
        model = core.compile_model(model, "NPU")
        input_layer_ir = model.input(0)
        output_layer_ir_0 = model.output(0)
        print("Resume model from %s" % opt.ckpt)
        # del checkpoint
    # else:
    #     print("[!] Retrain")
    #     model = nn.DataParallel(model)
    #     model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opt.crop_val:
        transform = T.Compose([
                T.Resize(opt.crop_size),
                T.CenterCrop(opt.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opt.save_val_results_to is not None:
        os.makedirs(opt.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        # model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            # img = img.to(device)
            
            # pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            pred = model([img])[output_layer_ir_0].max(1)[0]
            # import pdb
            # pdb.set_trace()
            
            colorized_preds = decode_fn(pred).astype('uint8')
            # colorized_preds = colorized_preds.squeeze()
            colorized_preds = Image.fromarray(colorized_preds)
            if opt.save_val_results_to:
                colorized_preds.save(os.path.join(opt.save_val_results_to, img_name+'.png'))



if __name__ == "__main__":
    opt = get_args()
    test(opt)
