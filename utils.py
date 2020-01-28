import time
import torch
import pickle
import torchvision.transforms as t
import torch.utils.data as data
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import h5py
import random
from shutil import copyfile
import io

class TextLogger():
    def __init__(self, title, save_path, append=False):
        print(save_path)
        file_state = 'wb'
        if append:
            file_state = 'ab'
        self.file = open(save_path, file_state, 0)
        self.log(title)
    
    def log(self, strdata):
        outstr = strdata + '\n'
        outstr = outstr.encode("utf-8")
        self.file.write(outstr)

    def __del__(self):
        self.file.close()

rot_transform = t.Compose([
    t.Resize((896, 896)),
    t.CenterCrop(448),
    #t.Resize((108, 108)),
    #t.Pad(12, padding_mode='reflect'),
    # t.Resize((96, 96)),
    # t.Pad(12, padding_mode='reflect'),
    #t.RandomCrop(96),
    # t.RandomCrop(96),
    # t.RandomHorizontalFlip(0.5),
    # t.RandomRotation([0, 360]),
    # t.ColorJitter(
    #     hue= 0.4,
    #     saturation=0.4,
    #     brightness=0.4,
    #     contrast=0.4),
    t.ToTensor(),
    ])

train_transform = t.Compose([
    t.Resize((896, 896)),
    t.CenterCrop(448),
    #t.Resize((108, 108)),
    #t.Pad(12, padding_mode='reflect'),
    # t.Resize((96, 96)),
    # t.Pad(12, padding_mode='reflect'),
    #t.RandomCrop(96),
    # t.RandomCrop(96),
    t.RandomHorizontalFlip(0.5),
    t.RandomRotation([0, 360]),
    t.ColorJitter(
        hue= 0.4,
        saturation=0.4,
        brightness=0.4,
        contrast=0.4),
    t.ToTensor(),
    ])
test_transform = t.Compose([
    t.Resize((896, 896)),
    t.CenterCrop(448),
    # t.Resize((96, 96)),
    #t.Resize((224, 224)),
    t.ToTensor(),
    ])


class ImageDataset_BACH(data.Dataset):
    def __init__(self, dataset_path, train, is_test, ss_task):

        self.train = train
        self.is_test = is_test
        self.ss_task = ss_task

        target = dataset_path #'mnt/datasets/bach/'
        # target = '/mnt/data/kupfersk/'
        train_x_path = 'bach_train.h5'
        train_y_path = 'bach_train.h5'
        valid_x_path = 'bach_val.h5'
        valid_y_path = 'bach_val.h5'
        test_x_path = 'bach_test.h5'
        test_y_path = 'bach_test.h5'

        if self.train == True:
            if self.ss_task == "none":
                self.transform = train_transform
            if self.ss_task == "rotation":
                self.transform = rot_transform

            self.h5_file_x = target + train_x_path  # '../../dataset/pcam/camelyonpatch_level_2_split_train_x.h5'
            self.h5_file_y = target + train_y_path  # '../../dataset/pcam/camelyonpatch_level_2_split_train_y.h5'

        else:
            if self.is_test == False:
                self.transform = test_transform
                self.h5_file_x = target + valid_x_path  # '../../dataset/pcam/camelyonpatch_level_2_split_valid_x.h5'
                self.h5_file_y = target + valid_y_path  # '../../dataset/pcam/camelyonpatch_level_2_split_valid_y.h5'
            else:
                self.transform = test_transform
                self.h5_file_x = target + test_x_path  # '../../dataset/pcam/camelyonpatch_level_2_split_test_x.h5'
                self.h5_file_y = target + test_y_path  # '../../dataset/pcam/camelyonpatch_level_2_split_test_y.h5'

        y_f = h5py.File(self.h5_file_y, 'r')
        if self.is_test == True:
            self.label = torch.Tensor(y_f['f']).squeeze()
        else:
            self.label = torch.Tensor(y_f['y']).squeeze()
        self.random_ixs = list(range(len(self.label)))
        random.shuffle(self.random_ixs)
        y_f.close()

        pil2tensor = t.ToTensor()
        self.data = h5py.File(self.h5_file_x, 'r')

        if not os.path.exists('data'):
            os.makedirs('data')

        if not os.path.exists('data/mean_std.pt'):
            mean_std = {}
            mean_std['mean'] = [0, 0, 0]
            mean_std['std'] = [0, 0, 0]
            x_f = h5py.File('../../dataset/pcam/camelyonpatch_level_2_split_train_x.h5')
            y_f = h5py.File('../../dataset/pcam/camelyonpatch_level_2_split_train_y.h5')
            labels = torch.Tensor(y_f['y']).squeeze()
            y_f.close()

            print('Calculating mean and std')
            for ix in tqdm(range(len(labels))):
                np_dat = x_f['x'][ix]
                img = pil2tensor(Image.fromarray(np_dat))
                for cix in range(3):
                    mean_std['mean'][cix] += img[cix, :, :].mean()
                    mean_std['std'][cix] += img[cix, :, :].std()

            for cix in range(3):
                mean_std['mean'][cix] /= len(labels)
                mean_std['std'][cix] /= len(labels)

            torch.save(mean_std, 'data/mean_std.pt')

        else:
            mean_std = torch.load('data/mean_std.pt')

        self.transform.transforms.append(t.Normalize(mean=mean_std['mean'], std=mean_std['std']))
        self.data.close()
        self.data = None

        self.I = list(range(len(self.label)))

    def __getitem__(self, index):

        if self.data == None:
            self.data = h5py.File(self.h5_file_x, 'r')
        curr_index = self.I[index]
        img = Image.open(io.BytesIO(self.data['x'][curr_index]))
        target = self.label[index]
        target = int(target.item())

        if self.ss_task == "rotation":
            rot_opt = [0, 90, 180, 270]
            target = random.randrange(4)
            img = t.functional.rotate(img, rot_opt[target])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return (len(self.label))


class ImageDataset_pcam(data.Dataset):
    def __init__(self, dataset_path, train, is_test, ss_task):

        self.train = train
        self.is_test = is_test
        self.ss_task = ss_task

        target = dataset_path#'/mnt/datasets/pcam/'
        train_x_path = 'camelyonpatch_level_2_split_train_x.h5'
        train_y_path = 'camelyonpatch_level_2_split_train_y.h5'
        valid_x_path = 'camelyonpatch_level_2_split_valid_x.h5'
        valid_y_path = 'camelyonpatch_level_2_split_valid_y.h5'
        test_x_path = 'camelyonpatch_level_2_split_test_x.h5'
        test_y_path = 'camelyonpatch_level_2_split_test_y.h5'


        if self.train == True:
            if self.ss_task == "none":
                self.transform = train_transform
            if self.ss_task == "rotation":
                self.transform = rot_transform

            self.h5_file_x = target + train_x_path#'../../dataset/pcam/camelyonpatch_level_2_split_train_x.h5'
            self.h5_file_y = target + train_y_path#'../../dataset/pcam/camelyonpatch_level_2_split_train_y.h5'
        else:
            if self.is_test==False:
                self.transform = test_transform
                self.h5_file_x = target + valid_x_path  # '../../dataset/pcam/camelyonpatch_level_2_split_valid_x.h5'
                self.h5_file_y = target + valid_y_path  # '../../dataset/pcam/camelyonpatch_level_2_split_valid_y.h5'
            else:
                self.transform = test_transform
                self.h5_file_x = target + test_x_path#'../../dataset/pcam/camelyonpatch_level_2_split_test_x.h5'
                self.h5_file_y = target + test_y_path#'../../dataset/pcam/camelyonpatch_level_2_split_test_y.h5'


        y_f = h5py.File(self.h5_file_y, 'r')
        self.label = torch.Tensor(y_f['y']).squeeze()
        self.random_ixs = list(range(len(self.label)))
        random.shuffle(self.random_ixs)
        y_f.close()

        pil2tensor = t.ToTensor()
        self.data = h5py.File(self.h5_file_x, 'r')

        if not os.path.exists('data'):
            os.makedirs('data')

        if not os.path.exists('data/mean_std.pt'):
            mean_std = {}
            mean_std['mean'] = [0,0,0]
            mean_std['std'] = [0,0,0]
            x_f = h5py.File('../../dataset/pcam/camelyonpatch_level_2_split_train_x.h5')
            y_f = h5py.File('../../dataset/pcam/camelyonpatch_level_2_split_train_y.h5')
            labels = torch.Tensor(y_f['y']).squeeze()
            y_f.close()

            print('Calculating mean and std')
            for ix in tqdm(range(len(labels))):
                np_dat = x_f['x'][ix]
                img = pil2tensor(Image.fromarray(np_dat))
                for cix in range(3):
                    mean_std['mean'][cix] += img[cix,:,:].mean()
                    mean_std['std'][cix] += img[cix,:,:].std()

            for cix in range(3):
                mean_std['mean'][cix] /= len(labels)
                mean_std['std'][cix] /= len(labels)

            torch.save(mean_std, 'data/mean_std.pt')

        else:
            mean_std = torch.load('data/mean_std.pt')


        self.transform.transforms.append(t.Normalize(mean=mean_std['mean'], std=mean_std['std']))
        self.data.close()
        self.data = None

    def __getitem__(self, index):
        
        if self.data == None:
            self.data = h5py.File(self.h5_file_x, 'r')
        img = Image.fromarray(self.data['x'][index])
        target = self.label[index]

        if self.ss_task == "rotation":
            rot_opt = [0,90,180,270]
            target = random.randrange(4)
            img = t.functional.rotate(img, rot_opt[target])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        # return int(0.01*len(self.label))
        return (len(self.label))


if __name__ == '__main__':
    train_logger = TextLogger('Train loss', 'train_loss.log')
    for ix in range(30):
        print(ix)
        train_logger.log('%s, %s' % (str(torch.rand(1)[0]), str(torch.rand(1)[0])))
        time.sleep(1)


