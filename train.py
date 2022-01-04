import torch
import os
import colorize_data_lab as D
#import basic_model
from torch.utils.data import DataLoader,Dataset
import argparse
import misc
import time

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

parser = argparse.ArgumentParser(description='PyTorch Colorize Training')
parser.add_argument('--n_batches', default=8, type=int, help='batch size')
parser.add_argument('--n-epochs', default=100, type=int, help='total epochs to run train')

parser.add_argument('--path-train', default='/home/choi574/Documents/samsung_internship_challenge/landscape_images_train/', 
        type=str, help='path to training data')
parser.add_argument('--path-val', default='/home/choi574/Documents/samsung_internship_challenge/landscape_images_val/', 
        type=str, help='path to validation data')

parser.add_argument('--resume', default=False, 
        action='store_true', help='True: resume training, False: start from scratch')
parser.add_argument('--mode', default='train', 
        type=str, help='val: run validation, train: run training, inference: run inference on one image')
parser.add_argument('--path-trained-model', default='./trained_model/', 
        type=str, help='path to save trained models')

parser.add_argument('--saved-epoch', default=0, type=int, help='epoch')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

parser.add_argument('--path-inf-img', default='./test_img.jpg', 
        type=str, help='path to image to inference')
parser.add_argument('--path-inf-img-out', default='./colorized_img', 
        type=str, help='path to save colorized image after inference')
parser.add_argument('--device', default='GPU', 
        type=str, help='CPU or GPU')


def main():
    args = parser.parse_args()

    if not os.path.isdir('./plots'):
        os.makedirs('./plots')
    if not os.path.isdir('./results'):
        os.makedirs('./results')
    if not os.path.isdir('./trained_model'):
        os.makedirs('./trained_model')
    if not os.path.isdir(args.path_trained_model):
        os.makedirs(args.path_trained_model)
    
    trainer = Trainer(args)
    trainer.train()


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    Args:
        optimizer: pytorch optimizer
        epoch: current training epoch 
        args: arguments
    Return:
        lr: updated learning rate
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class Trainer:
    def __init__(self, args):
        # Define hparams here or load them from a config file
        self.args = args

    def train(self):
        # dataloaders
        train_dataset = D.ColorizeData(self.args.path_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.n_batches, shuffle=True)
        val_dataset = D.ColorizeData(self.args.path_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.n_batches, shuffle=False)

        # Model
        #model = basic_model.Net()
        body = create_body(resnet18, pretrained=True, n_in=1, cut=-4)
        model = DynamicUnet(body, 2, (256,256))
        if self.args.device == 'GPU': model = model.cuda()
        
        # Loss function to use
        criterion = torch.nn.MSELoss()
        #criterion = torch.nn.L1Loss()
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        
        model.train()

        epoch_resume = 0
        if self.args.resume or self.args.mode=='val' or self.args.mode=='inference':
            fn_ckpt = os.path.join(self.args.path_trained_model+'model_color.pth')
            if not os.path.exists(fn_ckpt):
                print('[ERROR]: Cannot find Checkpoint file here: ', fn_ckpt)
                return
            print('Load pretrained model')
            trained_dict = torch.load(fn_ckpt)
            model.load_state_dict(trained_dict['model_params'])
            optimizer.load_state_dict(trained_dict['optim_params'])
            epoch_resume = trained_dict['epoch']

        if self.args.mode == 'val':
            self.validate(model, criterion, val_dataloader, epoch_resume)
            return
        elif self.args.mode == 'inference':
            self.inference(model, criterion)
            return

        loss_min = None
        
        # train loop
        for epoch in range(epoch_resume, self.args.n_epochs):
            track_loss = misc.AverageMeter('Loss')
            track_time_batch = misc.AverageMeter('TimeBatch')
            track_time_data = misc.AverageMeter('TimeData')
            end = time.time()
            lrl = adjust_learning_rate(optimizer, epoch, self.args)
            
            for batch_i, data in enumerate(train_dataloader):
                img_in, img_target, img_origin = data
                if self.args.device == 'GPU': img_in, img_target = img_in.cuda(), img_target.cuda()
                track_time_data.update(time.time() - end)


                # Prediction Forward
                pred = model(img_in)

                # Calculate Loss
                loss = criterion(pred, img_target)
                # Track loss
                track_loss.update(loss.item(), img_in.size(0))

                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track computation time
                track_time_batch.update(time.time() - end)

                if batch_i%100==0:
                    print('Epoch: {},  Batch: {}/{} \t Loss: {:.4f}, \t Time Data: {:.4f}, \t Compute: {:.4f}'.format(
                        epoch, batch_i, len(train_dataloader), track_loss.avg, track_time_data.avg, track_time_batch.avg))
                    fd = open('log_train', 'a')
                    fd.write('Epoch: {},  Batch: {}/{} \t Loss: {:.4f}, \t Time Data: {:.4f}, \t Compute: {:.4f}\n'.format(
                        epoch, batch_i, len(train_dataloader), track_loss.avg, track_time_data.avg, track_time_batch.avg))

                    if batch_i==0 and False:
                        bs = img_in.size(0)
                        misc.plot_samples_from_images(img_in, bs, './plots', 'gray_e{}_train'.format(epoch))
                        misc.plot_samples_from_images_lab(img_in, pred.detach(), bs, './plots', 'pred_e{}_train'.format(epoch))
                        misc.plot_samples_from_images_lab(img_in, img_target, bs, './plots', 'target_e{}_train'.format(epoch))
                        misc.plot_samples_from_images(img_origin, bs, './plots', 'origin_e{}_train'.format(epoch))
                        print('saved')

                end = time.time()


            if epoch%1 == 0:
                loss_val = self.validate(model, criterion, val_dataloader, epoch)
                model.train()
                
                if loss_min is None:
                    loss_min = loss_val

                if loss_val < loss_min:
                    loss_min = loss_val
                    checkpoint = {
                            "epoch": epoch, 
                            "model_params": model.state_dict(), 
                            "optim_params": optimizer.state_dict(), 
                            }
                    torch.save(checkpoint, os.path.join(self.args.path_trained_model, 'model_color.pth'))
                    print('Model is saved')


    def inference(self, model, criterion):
        model.eval()
        # Validation loop begin
        # ------
        end = time.time()

        # Load image
        img_in = D.load_gray_image(self.args.path_inf_img)
        if self.args.device=='GPU': img_in = img_in.cuda() 
        
        with torch.no_grad():

            # Prediction Forward
            pred = model(img_in)

            bs = img_in.size(0)
            misc.plot_samples_from_images(img_in, bs, './results', 'gray_input')
            misc.plot_samples_from_images_lab(img_in, pred.detach(), bs, './results', self.args.path_inf_img_out)

            print('[Inference]: Colorized image is saved')

        return 


    def validate(self, model, criterion, val_dataloader, epoch):
        model.eval()
        # Validation loop begin
        # ------
        track_loss = misc.AverageMeter('Loss')
        track_time_batch = misc.AverageMeter('TimeBatch')
        track_time_data = misc.AverageMeter('TimeData')
        end = time.time()
        
        with torch.no_grad():
            for batch_i, data in enumerate(val_dataloader):
                img_in, img_target, img_origin = data
                if self.args.device == 'GPU': img_in, img_target = img_in.cuda(), img_target.cuda()
                track_time_data.update(time.time() - end)

                # Prediction Forward
                pred = model(img_in)

                # Calculate Loss
                loss = criterion(pred, img_target)
                # Track loss
                track_loss.update(loss.item(), img_in.size(0))

                # Track computation time
                track_time_batch.update(time.time() - end)
                end = time.time()

            print('Validation :: Epoch: {},  \t Loss: {:.4f}, \t Time Data: {:.4f}, \t Compute: {:.4f}'.format(
                epoch, track_loss.avg, track_time_data.avg, track_time_batch.avg))
            fd = open('log_val', 'a')
            fd.write('Validation :: Epoch: {},  Batch: {}/{} \t Loss: {:.4f}, \t Time Data: {:.4f}, \t Compute: {:.4f} \n'.format(
                epoch, batch_i, len(val_dataloader), track_loss.avg, track_time_data.avg, track_time_batch.avg))

            if False:
                bs = img_in.size(0)
                misc.plot_samples_from_images(img_in, bs, './plots', 'gray_e{}_val'.format(epoch))
                misc.plot_samples_from_images_lab(img_in, pred.detach(), bs, './plots', 'pred_e{}_val'.format(epoch))
                misc.plot_samples_from_images_lab(img_in, img_target, bs, './plots', 'target_e{}_val'.format(epoch))

        return loss.item()



if __name__ == "__main__":
    main()
