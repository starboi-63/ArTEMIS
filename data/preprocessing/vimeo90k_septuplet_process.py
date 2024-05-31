import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import torch
import numpy as np


class VimeoSeptuplet(Dataset):
    def __init__(self, data_dir, is_training):
        '''
        make a Vimeo Septuplet object

        data_dir :: root directory path for septuplet dataset from Vimeo
        is_training :: true for training, false for testing
        '''
        self.data_dir = data_dir
        # Vimeo90k dataset organizes images in a 'sequences' folder
        self.image_root = os.path.join(self.data_dir, 'sequences')
        self.training = is_training

        # par down sep_trainlist.txt files after downloading vimeo dataset
        train_fn = os.path.join(self.data_dir, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_dir, 'sep_testlist.txt')

        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
            
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        # data augmentation
        if self.training:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):  # dataset[index]
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])

        # septuplet indicies range 1-7
        imgpaths = [imgpath + f'/im{i}.png' for i in range(1, 8)]
        images = [Image.open(pth) for pth in imgpaths]

        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
            images_ = []
            for img_ in images:
                set_seed(seed)
                # augmentation, cropping & flipping
                images_.append(self.transforms(img_))
            images = images_

            # Random Temporal Flip
            # going 'forwards' or 'backwards' in a video should be the same
            # 50% chance that image data is given to you in the forwards or backwards order
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]

            # Randomly select a ground truth frame 
            random_index = random.randint(2, 4)
            ground_truth = images[random_index]

            # images --> solely the input frames w/o interpolated frames
            context = images[:2] + images[5:]

            match random_index:
                case 2:
                    output_frame_time = 0.25
                case 3:
                    output_frame_time = 0.5
                case 4:
                    output_frame_time = 0.75
                case _: # default frame time
                    output_frame_time = 0.5

            return context, ground_truth, output_frame_time

        else:
            images = [self.transforms(img_) for img_ in images]

            # Randomly select a ground truth frame 
            random_index = random.randint(2, 4)
            ground_truth = images[random_index]
            context = images[:2] + images[5:]

            match random_index:
                case 2:
                    output_frame_time = 0.25
                case 3:
                    output_frame_time = 0.5
                case 4:
                    output_frame_time = 0.75
                case _: # default frame time
                    output_frame_time = 0.5

            return context, ground_truth, output_frame_time

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)


def get_loader(mode, data_dir, batch_size, num_workers):
    # If just running a forward pass, no need to construct a DataLoader
    if mode not in ['train', 'test']:
        return None

    is_training = mode == 'train'
    dataset = VimeoSeptuplet(data_dir, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_training, num_workers=num_workers, pin_memory=True)


def set_seed(seed=None, cuda=True): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed_all(seed)
