
import os, argparse, time, subprocess, io, shlex, pickle, pprint
import pandas as pd
import numpy as np
import tqdm
import fire
import os
import torch
import pickle
import time
import tqdm
import numpy as np
import pprint
from PIL import Image
from torchvision import transforms
parser = argparse.ArgumentParser(description='ImageNet testing')
## General parameters
parser.add_argument('--in_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('-restore_epoch', '--restore_epoch', default=0, type=int,
                    help='epoch number for restoring model training ')
parser.add_argument('-restore_path', '--restore_path', default=None, type=str,
                    help='path of folder containing specific epoch file for restoring model training')

## Training parameters
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=20, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=70, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--optimizer', choices=['stepLR', 'plateauLR'], default='stepLR',
                    help='Optimizer')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=20, type=int,
                    help='after how many epochs learning rate should be decreased by step_factor')
parser.add_argument('--step_factor', default=0.1, type=float,
                    help='factor by which to decrease the learning rate')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')

## Model parameters
parser.add_argument('--torch_seed', default=0, type=int,
                    help='seed for weights initializations and torch RNG')
parser.add_argument('--model_arch', choices=['alexnet', 'resnet50', 'resnet50_at', 'cornets','cornetr','cornetalex'], default='cornetalex',
                    help='back-end model architecture to load')
parser.add_argument('--normalization', choices=['vonenet', 'imagenet'], default='vonenet',
                    help='image normalization to apply to models')
parser.add_argument('--visual_degrees', default=8, type=float,
                    help='Field-of-View of the model in visual degrees')

## VOneBlock parameters
# Gabor filter bank
parser.add_argument('--stride', default=4, type=int,
                    help='stride for the first convolution (Gabor Filter Bank)')
parser.add_argument('--ksize', default=25, type=int,
                    help='kernel size for the first convolution (Gabor Filter Bank)')
parser.add_argument('--simple_channels', default=256, type=int,
                    help='number of simple channels in V1 block')
parser.add_argument('--complex_channels', default=256, type=int,
                    help='number of complex channels in V1 block')
parser.add_argument('--gabor_seed', default=0, type=int,
                    help='seed for gabor initialization')
parser.add_argument('--sf_corr', default=0.75, type=float,
                    help='')
parser.add_argument('--sf_max', default=6, type=float,
                    help='')
parser.add_argument('--sf_min', default=0, type=float,
                    help='')
parser.add_argument('--rand_param', choices=[True, False], default=False, type=bool,
                    help='random gabor params')
parser.add_argument('--k_exc', default=25, type=float,
                    help='')

# Noise layer
parser.add_argument('--noise_mode', choices=['gaussian', 'neuronal', None],
                    default='neuronal',
                    help='noise distribution')
parser.add_argument('--noise_scale', default=0.35, type=float,
                    help='noise scale factor')
parser.add_argument('--noise_level', default=0.07, type=float,
                    help='noise level')


FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    if n > 0:
        gpus = subprocess.run(shlex.split(
            'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
            stdout=subprocess.PIPE).stdout
        gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
        gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            visible = [int(i)
                       for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            gpus = gpus[gpus['index'].isin(visible)]
        gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
from vonenet import get_model

torch.manual_seed(FLAGS.torch_seed)

torch.backends.cudnn.benchmark = True

if FLAGS.ngpus > 0:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = 'cpu'

if FLAGS.normalization == 'vonenet':
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
elif FLAGS.normalization == 'imagenet':
    print('Imagenet standard normalization')
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]


def load_model():
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    print('Getting VOneNet')
    model = get_model(map_location=map_location, model_arch=FLAGS.model_arch, pretrained=True,
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.simple_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc)

    if FLAGS.ngpus > 0 and torch.cuda.device_count() > 1:
        print('We have multiple GPUs detected')
        model = model.to(device)
    elif FLAGS.ngpus > 0 and torch.cuda.device_count() == 1:
        print('We run on GPU')
        model = model.to(device)
    else:
        print('No GPU detected!')
        model = model.module

    return model

from torch.nn import Module
class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model
def test():
    
    model = load_model()
    ckpt_data = torch.load('pretrain_model.pth.tar')
    model.load_state_dict(ckpt_data['state_dict'])
    validator=ImageNetVal(model)
    validator()


class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(device)

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.in_path, 'val'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                target = target.to(device)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)
        print(record)
        return record


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)


