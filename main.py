import time
from tqdm import tqdm
import config
import myutils
import shutil
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model.artemis import ArTEMIS
from torch.optim import Adamax
from loss import Loss
from data.preprocessing.vimeo90k_septuplet_process import get_loader


def load_checkpoint(args, model, optimizer, path):
    print("loading checkpoint %s" % path)
    checkpoint = torch.load(path)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint.get("lr", args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
save_location = os.path.join(args.checkpoint_dir, "checkpoints")

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)

if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

# Initialize Tensorboard
writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'tensorboard_logs'))

# Initialize DataLoaders
if args.dataset == "vimeo90K_septuplet":
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
else:
    raise NotImplementedError

print("Building model: %s" % args.model)

# Important parameters: num_inputs=4, num_outputs=3
# Model can calculate delta_t, the perceived timestep between each frame, including inputs and outputs
model = ArTEMIS(num_inputs=args.nbr_frame, joinType=args.joinType, kernel_size=args.kernel_size, dilation=args.dilation, num_outputs=args.num_outputs)
model = nn.DataParallel(model).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of network parameters: {}'.format(total_params))

##### Define Loss & Optimizer #####
criterion = Loss(args)

# TODO: Different learning rate schemes for different parameters
optimizer = Adamax(model.parameters(), lr=args.lr,
                   betas=(args.beta1, args.beta2))


def train(args, epoch):
    torch.cuda.empty_cache()
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.train()
    criterion.train()

    for i, (images, gt_images) in enumerate(train_loader):


        # Build input batch
        images = [img_.to(device) for img_ in images]

        # Forward
        optimizer.zero_grad()

        # out should be a list of the 3 interpolated frames
        out_ll, out_l, out = model(images)
        print("out length", len(out))
        
        # Temporally Flip inputs: going 'forwards' or 'backwards' in a video
        # reverse_out_ll, reverse_out_l, reverse_out = model(images[::-1])

        gt = [gt_image.to(device) for gt_image in gt_images]
        print("gt length", len(gt))

        # TODO: Adjust loss calculation for 3 frames
        # Option 1: Average 3 losses for each frame
        # Option 2: Have 3 losses for each frame to backprop

        # ********************************************************************************
        # need to also pass in temporally flipped interpolated frames to loss calculations
        loss0, _ = criterion(out[0], gt[0])#reverse_out[0], gt[0])
        loss1, _ = criterion(out[1], gt[1])#reverse_out[1], gt[1])
        loss2, _ = criterion(out[2], gt[2])#reverse_out[2], gt[2])
        overall_loss = (loss0 + loss1 + loss2) / 3

        losses['total'].update(overall_loss.item())

        overall_loss.backward()
        optimizer.step()

        # loss, _ = criterion(out, gt)
        # overall_loss = loss

        # losses['total'].update(loss.item())

        # overall_loss.backward()
        # optimizer.step()

        # Calc metrics & print logs
        if i % args.log_iter == 0:
            myutils.eval_metrics(out, gt, psnrs, ssims)

            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}  Lr:{:.6f}'.format(
                epoch, i, len(train_loader), losses['total'].avg, psnrs.avg, optimizer.param_groups[0]['lr'], flush=True))

            # Reset metrics
            losses, psnrs, ssims = myutils.init_meters(args.loss)


def test(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()
    torch.cuda.empty_cache()

    t = time.time()
    with torch.no_grad():
        for i, (images, gt_images, _) in enumerate(tqdm(test_loader)):

            images = [img_.to(device) for img_ in images]
            gt = gt_images.to(device)

            # images is a list of neighboring frames
            out = model(images)

            # Save loss values
            # loss, loss_specific = criterion(out, gt)

            # ********************************************************************************
            # need to also pass in temporally flipped interpolated frames to loss calculations
            loss0, loss_specific0 = criterion(out[0], gt[0])
            loss1, loss_specific1 = criterion(out[1], gt[1])
            loss2, loss_specific2 = criterion(out[2], gt[2])
            overall_loss = (loss0 + loss1 + loss2) / 3
            # TODO: not sure if loss_specific is done right...
            loss_specific = {
                'type': loss_specific0['type'],
                'weight': loss_specific0['weight'],
                'function': loss_specific0['function']
            }

            # Save loss values
            for k, v in losses.items():
                if k != 'total':
                    # TODO: idk what loss_specific does
                    v.update(loss_specific[k].item())
            losses['total'].update(overall_loss.item())

            # Evaluate metrics
            myutils.eval_metrics(out, gt, psnrs, ssims)

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
