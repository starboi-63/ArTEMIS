import time
from tqdm import tqdm
import config
import metrics
import shutil
import os

import lightning as L
import torch
import torch.nn as nn

from model.artemis import ArTEMIS
from torch.optim import Adamax
from loss import Loss
from metrics import eval_metrics
from data.preprocessing.vimeo90k_septuplet_process import get_loader


# Parse command line arguments
args, unparsed = config.get_args()
save_location = os.path.join(args.checkpoint_dir, "checkpoints")

# Initialize CUDA & set random seed
device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.random_seed)

if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

# Initialize DataLoaders
if args.dataset == "vimeo90K_septuplet":
    t0 = time.time()
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    t1 = time.time()
    print("Time to load train loader: ", t1-t0)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    t2 = time.time()
    print("Time to load test loader", t2-t1)
else:
    raise NotImplementedError


class ArTEMISModel(L.LightningModule):
    def __init__(self, cmd_line_args=args):
        super().__init__()
        # Call this to save command line arguments to checkpoints
        self.save_hyperparameters()
        # Initialize instance variables
        self.args = args
        self.model = ArTEMIS(num_inputs=args.nbr_frame, joinType=args.joinType, kernel_size=args.kernel_size, dilation=args.dilation, num_outputs=args.num_outputs)
        self.optimizer = Adamax(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.loss = Loss(args)
        self.metrics = eval_metrics

    def forward(self, images):
        return self.model(images)
    
    def training_step(self, batch, batch_idx):
        images, gt_images = batch
        outputs = self(images)
        loss = self.loss(outputs, gt_images)

        # log metrics for each step
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, gt_images = batch
        outputs = self.model(images)
        loss = self.loss(outputs, gt_images)
        psnr, ssim = self.metrics(outputs, gt_images)

        # log metrics for each step
        self.log('test_loss', loss)
        
    
    def configure_optimizers(self):
        return self.optimizer
    

# # call after training
# trainer = L.Trainer()
# trainer.fit(model=model, train_dataloaders=dataloader)

# # automatically auto-loads the best weights from the previous run
# trainer.test(dataloaders=test_dataloaders)

# # or call with pretrained model
# model = LightningTransformer.load_from_checkpoint(PATH)
# dataset = WikiText2()
# test_dataloader = DataLoader(dataset)
# trainer = L.Trainer()
# trainer.test(model, dataloaders=test_dataloader)

model = ArTEMISModel(args)
trainer = L.Trainer()

def train(args, epoch):

        # Calc metrics & print logs
        if i % args.log_iter == 0:
            for out_image, ground_truth_image in zip(out, gt):
                metrics.eval_metrics(out_image, ground_truth_image, psnrs, ssims)
    
            # Reset metrics
            losses, psnrs, ssims = metrics.init_meters(args.loss)


def test(args, epoch):

    losses, psnrs, ssims = metrics.init_meters(args.loss)

            loss_specific = {
                'type': loss_specific0['type'],
                'weight': loss_specific0['weight'],
                'function': loss_specific0['function']
            }

            # Save loss values
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())

            losses['total'].update(overall_loss.item())

            # Evaluate metrics
            for out_image, ground_truth_image in zip(out, gt):
                metrics.eval_metrics(out_image, ground_truth_image, psnrs, ssims)
            for out_image, ground_truth_image in zip(out, gt):
                metrics.eval_metrics(out_image, ground_truth_image, psnrs, ssims)

    return losses['total'].avg, psnrs.avg, ssims.avg



def print_log(epoch, num_epochs, one_epoch_time, oup_pnsr, oup_ssim, Lr):
    print('({0:.0f}s) Epoch [{1}/{2}], Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}'
          .format(one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim))
    # write training log
    with open('./training_log/train_log.txt', 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}, Lr:{6}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim, Lr), file=f)


lr_schular = [2e-4, 1e-4, 5e-5, 2.5e-5, 5e-6, 1e-6]
training_schedule = [40, 60, 75, 85, 95, 100]


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for i in range(len(training_schedule)):
        if epoch < training_schedule[i]:
            current_learning_rate = lr_schular[i]
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate
        print('Learning rate sets to {}.'.format(param_group['lr']))


""" Entry Point """


def main(args):
    # load_checkpoint(args, model, optimizer, save_location+'/epoch20/model_best.pth')
    # test_loss, psnr, ssim = test(args, args.start_epoch)
    # print(psnr)

    best_psnr = 0

    for epoch in range(args.start_epoch, args.max_epoch):
        adjust_learning_rate(optimizer, epoch)
        start_time = time.time()
        train(args, epoch)

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.param_groups[-1]['lr']
        }, os.path.join(save_location, 'checkpoint.pth'))

        test_loss, psnr, ssim = test(args, epoch)

        # save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)

        if is_best:
            shutil.copyfile(os.path.join(save_location, 'checkpoint.pth'),
                            os.path.join(save_location, 'model_best.pth'))

        one_epoch_time = time.time() - start_time
        print_log(epoch, args.max_epoch, one_epoch_time, psnr,
                  ssim, optimizer.param_groups[-1]['lr'])


if __name__ == "__main__":
    main(args)
