import torch
import VGG
from model import *
from datasets import *
from torchvision.utils import save_image,make_grid
import os
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def test(args):
    # parameters
    cont_img_path = args.cont_img_path
    style_img_path = args.style_img_path
    model_checkpoint = args.model_checkpoint
    vgg_checkpoint = args.vgg_checkpoint
    output_file = args.output_file

    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((512, 512), Image.BICUBIC),
        transforms.ToTensor()
    ])
    cont_img = transform(Image.open(cont_img_path).convert('RGB'))
    style_img = transform(Image.open(style_img_path).convert('RGB'))
    low_cont = resize(cont_img, cont_img.shape[-1] // 2)
    low_style = resize(style_img, style_img.shape[-1] // 2)

    # initialize model and optimizer
    vgg = VGG.vgg
    vgg.load_state_dict(torch.load(vgg_checkpoint))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    net = VGG.Net(vgg).to(device)

    model = Model().to(device)
    model.load_state_dict(torch.load(model_checkpoint))

    cont_img = cont_img.to(device)
    low_cont = low_cont.to(device)
    low_style = low_style.to(device)
    model.eval()
    cont_feat = net.encode_with_intermediate(low_cont.unsqueeze(0))
    style_feat = net.encode_with_intermediate(low_style.unsqueeze(0))

    coeffs, output = model(cont_img.unsqueeze(0), cont_feat, style_feat)

    save_image(output, output_file + 'output.jpg', normalize=True)
    return

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Joint Bilateral learning')
    parser.add_argument('--cont_img_path', type=str, required=True, help='path to content images')
    parser.add_argument('--style_img_path', type=str, required=True, help='path to style images')
    parser.add_argument('--vgg_checkpoint', type=str, default="./checkpoints/vgg_normalised.pth",
                        help='path to style images')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='path to style images')
    parser.add_argument('--output_file', type=str, default='./output/')

    params = parser.parse_args()

    print('PARAMS:')
    print(params)

    test(params)