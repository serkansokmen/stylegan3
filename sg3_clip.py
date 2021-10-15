## Generates images using nvidia stylegan3 with CLIP guidance.
# Sampling tricks and some code thanks to Katherine Crowson (https://twitter.com/RiversHaveWings)
# Basic changes by Nerdy Rodent for running locally
# Original Colab - https://colab.research.google.com/drive/1eYlenR1GHPZXt-YuvXabzO9wfh9CWY36

# Licensed under the MIT License
# Copyright (c) 2021 nshepperd; Katherine Crowson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# N/a - download/install for colab
#pip install --upgrade torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
#git clone https://github.com/NVlabs/stylegan3
#git clone https://github.com/openai/CLIP = pip install git+https://github.com/openai/CLIP.git
#pip install -e ./CLIP
#pip install einops ninja

# Imports
import sys
#sys.path.append('./CLIP')
#sys.path.append('./stylegan3')

import io
import os, time
import pickle
import shutil
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
torch.backends.cudnn.benchmark = True

#from torch_optimizer import DiffGrad

import clip
import requests
from PIL import Image
from einops import rearrange
import re
from subprocess import Popen, PIPE, run
#from IPython.display import display
#from google.colab import files

# Create the parser
vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

# Add the arguments

# General
vq_parser.add_argument("-t",  "--text_prompt",  type=str,   help="Text prompt", default='A nerdy rodent', dest='text_prompt')
#vq_parser.add_argument("-i", "--image_prompt", type=str,   help="Image prompt", default=None, dest='image_prompt')
vq_parser.add_argument("-n",  "--network",      type=str,   help="StyleGAN3 pre-trained network pkl location", default=None, dest='network_url')
vq_parser.add_argument("-d",  "--device",       type=str,   help="CUDA device", default='cuda:0', dest='cuda_device')
vq_parser.add_argument("-s",  "--steps",        type=int,   help="Steps", default=150, dest='steps')

# Model / Cutouts
vq_parser.add_argument("-p",  "--psi",          type=float, help="psi", default=0.7, dest='psi')
vq_parser.add_argument("-l",  "--lr",           type=float, help="Learning rate", default=0.03, dest='lr')
vq_parser.add_argument("-cp", "--cut_power",    type=float, help="Cut power", default=0.5, dest='cutp')
vq_parser.add_argument("-cn", "--cut_number",   type=int,   help="Number of cuts", default=32, dest='cutn')
#vq_parser.add_argument("-m", "--model",        type=str,   help="CLIP model", default='ViT-B/32', dest='clip_model')

# Option
vq_parser.add_argument("-a",  "--alternate",    action='store_true', help="Use alternate init method", dest='alt_init_method')

# video
vq_parser.add_argument("-if", "--input_fps",    type=int,   help="Input FPS", default=30, dest='input_fps')
vq_parser.add_argument("-of", "--output_fps",   type=int,   help="Output FPS", default=30, dest='output_fps')
vq_parser.add_argument("-f",  "--filename",     type=str,   help="Video filename", default='output.mp4', dest='video_filename')

# Execute the parse_args() method
args = vq_parser.parse_args()

if not args.network_url:
    print("Missing network")
    sys.exit(1)


# Functions
def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def fetch_model(url_or_path):
    if os.path.exists(url_or_path):
        return url_or_path
    else:
        basename = os.path.basename(url_or_path)
        try:
            os.system("wget -c '{url_or_path}'")
        except FileNotFoundError:
            print("wget not found - cannot download from given URL.")
            
        return basename

def norm1(prompt):
    "Normalize to the unit sphere."
    return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

def embed_image(image):
    n = image.shape[0]
    cutouts = make_cutouts(image)
    embeds = clip_model.embed_cutout(cutouts)
    embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
    return embeds

def embed_url(url):
    image = Image.open(fetch(url)).convert('RGB')
    return embed_image(TF.to_tensor(image).to(device).unsqueeze(0)).mean(0).squeeze(0)

class CLIP(object):
    def __init__(self):
        clip_model = "ViT-B/32"
        self.model, _ = clip.load(clip_model)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def embed_text(self, prompt):
        "Normalized clip text embedding."
        return norm1(self.model.encode_text(clip.tokenize(prompt).to(device)).float())

    def embed_cutout(self, image):
        "Normalized clip image embedding."
        return norm1(self.model.encode_image(self.normalize(image)))
  

# Setup and user output
device = torch.device(args.cuda_device)
print('Using device:', device, file=sys.stderr)

make_cutouts = MakeCutouts(224, args.cutn, args.cutp)
clip_model = CLIP()

# Load stylegan model
#base_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"
#model_name = "stylegan3-t-ffhqu-1024x1024.pkl"
#network_url = base_url + model_name

network_url = args.network_url
with open(fetch_model(network_url), 'rb') as fp:
    G = pickle.load(fp)['G_ema'].to(device)

# # Fix the coordinate grid to w_avg
# shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0))
# G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
# G.synthesis.input.affine.weight.data.zero_()

# # Arbitrary coordinate grid (dubious idea)
# with torch.no_grad():
#   grid = G.synthesis.input(G.mapping.w_avg.unsqueeze(0))
#   def const(x):
#     def f(w):
#       n = w.shape[0]
#       return x.broadcast_to([n, *x.shape[1:]])
#     return f
#   G.synthesis.input.forward = const(grid)
# grid.requires_grad_()

zs = torch.randn([10000, G.mapping.z_dim], device=device)
w_stds = G.mapping(zs, None).std(0)


# Run Settings

#target = embed_url("https://4.bp.blogspot.com/-uw859dFGsLc/Va5gt-bU9bI/AAAAAAAA4gM/dcaWzX0ZxdI/s1600/Lubjana+dragon+1.jpg")
#target = embed_url("https://irc.zlkj.in/uploads/e399d2fee2c6edd9/20210827165231_0_nexus%20of%20abandoned%20places.%20trending%20on%20ArtStation.png")
#seed = 2

target = clip_model.embed_text(args.text_prompt)
steps = args.steps

# Actually do the run
tf = Compose([
    Resize(224),
    lambda x: torch.clamp((x+1)/2,min=0,max=1),
    ])

def run():
    timestring = time.strftime('%Y%m%d%H%M%S')
    
    # Just go with a random seed every time for now
    seed = None;
    if seed is None:
        seed = torch.seed()
    
    torch.manual_seed(seed)
    print("Seed:", seed)
    
    # Init
    if not args.alt_init_method:
        # Method 1: sample 32 inits and choose the one closest to prompt
        with torch.no_grad():
            qs = []
            losses = []
            
            for _ in range(8):
                q = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None, truncation_psi=args.psi) - G.mapping.w_avg) / w_stds
                images = G.synthesis(q * w_stds + G.mapping.w_avg)
                embeds = embed_image(images.add(1).div(2))
                loss = spherical_dist_loss(embeds, target).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
          
            qs = torch.stack(qs)
            losses = torch.stack(losses)
            print(losses)
            print(losses.shape, qs.shape)
            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0).requires_grad_()
    else:
        # Method 2: Random init depending only on the seed.
        q = (G.mapping(torch.randn([1,G.mapping.z_dim], device=device), None, truncation_psi=args.psi) - G.mapping.w_avg) / w_stds
        q.requires_grad_()

    # Sampling loop
    q_ema = q
    opt = torch.optim.AdamW([q], lr=args.lr, betas=(0.0,0.999))
    #opt = DiffGrad([q], lr=args.lr)
    
    loop = tqdm(range(steps))
    for i in loop:
        opt.zero_grad()
        w = q * w_stds
        image = G.synthesis(w + G.mapping.w_avg, noise_mode='const')
        embed = embed_image(image.add(1).div(2))
        loss = spherical_dist_loss(embed, target).mean()
        loss.backward()
        opt.step()
        loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())

        q_ema = q_ema * 0.9 + q * 0.1
        image = G.synthesis(q_ema * w_stds + G.mapping.w_avg, noise_mode='const')

        # Just making videos, so save every frame
        if i % 1 == 0:
            #display(TF.to_pil_image(tf(image)[0]))          
            pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
            os.makedirs(f'samples/{timestring}', exist_ok=True)
            pil_image.save(f'samples/{timestring}/{i:04}.png')
  
    # Save images as a tar archive
    #!tar cf samples/{timestring}.tar samples/{timestring}
    #if os.path.isdir('drive/MyDrive/samples'):
    #    shutil.copyfile(f'samples/{timestring}.tar', f'drive/MyDrive/samples/{timestring}.tar')
    #else:
    #    files.download(f'samples/{timestring}.tar')

    # Create video
    init_frame = 0
    last_frame = i+1
    
    input_fps = args.input_fps
    output_fps = args.output_fps
    m_filename = args.video_filename

    frames = []
    print('Generating video...')
    for k in range(init_frame,last_frame):
        temp = Image.open(f'samples/{timestring}/{k:04}.png')
        keep = temp.copy()
        frames.append(keep)
        temp.close()
    
    print("Creating video...")
    ffmpeg_failed = False
    ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:fps={output_fps}'"
    output_file = re.compile('\.png$').sub('.mp4', m_filename)            
    try:
        p = Popen(['ffmpeg',
                   '-y',
                   '-f', 'image2pipe',
                   '-vcodec', 'png',
                   '-r', str(input_fps),
                   '-i',
                   '-',
                   '-vcodec', 'libx264',
                   '-r', str(output_fps),
                   '-pix_fmt', 'yuv420p',
                   '-crf', '17',
                   '-preset', 'veryslow',
                   '-filter:v', f'{ffmpeg_filter}',
                   output_file], stdin=PIPE)
    except FileNotFoundError:
        print("Can't open ffmpeg to create video. Is ffmpeg installed and in the path?")
        ffmpeg_failed = True
    
    if not ffmpeg_failed:    
        for im in tqdm(frames):
            im.save(p.stdin, 'PNG')
        p.stdin.close()
        p.wait()    

try:
    run()
except KeyboardInterrupt:
    pass
    
torch.cuda.empty_cache()
