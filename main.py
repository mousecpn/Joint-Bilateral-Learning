import torch
import VGG
from model import *
from datasets import *
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torchvision.utils import save_image,make_grid
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def sample_image(vgg, model,batch,epoch,output_file):
    cont_img,low_cont,style_img,low_style = batch
    batch_size = cont_img.shape[0]
    model.eval()
    cont_feat = vgg.encode_with_intermediate(low_cont)
    style_feat = vgg.encode_with_intermediate(low_style)
    coeffs, output = model(cont_img, cont_feat, style_feat)

    cont = make_grid(cont_img, nrow=batch_size, normalize=True)
    style = make_grid(style_img, nrow=batch_size, normalize=True)
    out = make_grid(output, nrow=batch_size, normalize=True)

    image_grid = torch.cat((cont, style, out), 1)
    save_image(image_grid, output_file + 'output'+str(epoch)+'.jpg', normalize=False)
    model.train()
    return

def train(args):
    cont_img_path = args.cont_img_path
    style_img_path = args.style_img_path
    batch_size = args.batch_size
    vgg_checkpoint = args.vgg_checkpoint
    output_file = args.output_file
    log_interval = args.log_interval
    ckpt_interval = args.ckpt_interval
    # set dataset
    device = torch.device("cuda")
    train_dataset = JBLDataset(cont_img_path, style_img_path, img_size=512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # initialize model and optimizer
    vgg = VGG.vgg
    vgg.load_state_dict(torch.load(vgg_checkpoint))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    net = VGG.Net(vgg).to(device)

    model = Model().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    L_loss = LaplacianRegularizer()
    epochs = 100
    batch_done = 0
    # training iteration
    for e in range(epochs):
        model.train()
        for i ,(low_cont, cont_img,style_img,low_style) in enumerate(train_loader):
            optimizer.zero_grad()

            cont_img = cont_img.to(device)
            low_cont = low_cont.to(device)
            style_img = style_img.to(device)
            low_style = low_style.to(device)

            cont_feat = net.encode_with_intermediate(low_cont)
            style_feat = net.encode_with_intermediate(low_style)

            coeffs,output = model(cont_img,cont_feat,style_feat)

            loss_c,loss_s  = net.loss(output,cont_img,style_img)
            loss_r = L_loss(coeffs)

            total_loss = 0.5 * loss_c + loss_s + 0.15 * loss_r

            total_loss.backward()
            optimizer.step()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [C loss: %f, S loss: %f, R loss: %f, total loss: %f]"
                % (
                    e,
                    epochs,
                    i,
                    len(train_loader),
                    loss_c.item(),
                    loss_s.item(),
                    loss_r.item(),
                    total_loss.item(),
                )
            )
            if (batch_done+1) % log_interval == 0:
                batch = [cont_img,low_cont,style_img,low_style]
                sample_image(net, model, batch, e,output_file)

            if (batch_done + 1) % ckpt_interval == 0:
                model.eval().cpu()
                ckpt_model_filename = "ckpt_" + str(e) + '_' + str(batch_done) + ".pth"
                ckpt_model_path = os.path.join('./checkpoints', ckpt_model_filename)
                torch.save(model.state_dict(), ckpt_model_path)
                model.to(device).train()
            batch_done += 1
    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Joint Bilateral learning')
    parser.add_argument('--cont_img_path', type=str, default="./data/content", help='path to content images')
    parser.add_argument('--style_img_path', type=str, default="./data/style", help='path to style images')
    parser.add_argument('--vgg_checkpoint', type=str, default="./checkpoints/vgg_normalised.pth", help='path to style images')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', type=str, default='./output/')
    parser.add_argument('--log_interval', type=int, default=600)
    parser.add_argument('--ckpt_interval', type=int, default=600)

    params = parser.parse_args()

    print('PARAMS:')
    print(params)

    # cont_img_path = "/home/dailh/VOC2007/JEPGImages"
    # style_img_path = "/home/dailh/pytorch-multiple-style-transfer-master/water_quality"
    # batch_size = 8
    # vgg_checkpoint = "/home/dailh/Joint-Bilateral-Learning/checkpoints/vgg_normalised.pth"
    # output_file = './output/'
    # log_interval = 600
    # ckpt_interval = 600
    train(params)