import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import os.path as osp
from utils.data_io import *
from utils.utils import *
import cv2
from torch.utils.data import DataLoader

# 1900 * 1060

class NTIRE_Training_Dataset(Dataset):

    def __init__(self, root_dir,crop=False, crop_size=None):
        self.root_dir = root_dir
        self.crop=crop
        self.crop_size=crop_size

        self.scenes_dir = os.path.join(self.root_dir, 'training_crop')
        self.scenes_dir_list = os.listdir(self.scenes_dir)
        # print(self.scenes_dir_list)
        # print(self.crop_dir_list)

        # dataset
        self.image_list = []

        for scene in range(len(self.scenes_dir_list)):
            exposures_path = osp.join(self.scenes_dir, self.scenes_dir_list[scene], 'exposures.npy')
            align_ratio_path = osp.join(self.scenes_dir, self.scenes_dir_list[scene], 'alignratio.npy')
            # image_path = sorted(glob.glob(os.path.join(self.scenes_dir, self.scenes_dir_list[scene], '*.png')))

            gt_img = os.path.join(self.scenes_dir, self.scenes_dir_list[scene],  'gt.png')
            short_img = os.path.join(self.scenes_dir, self.scenes_dir_list[scene],
                                     'short.png')
            med_img = os.path.join(self.scenes_dir, self.scenes_dir_list[scene],
                                   'medium.png')
            long_img = os.path.join(self.scenes_dir, self.scenes_dir_list[scene],
                                    'long.png')

            image_path = [gt_img, long_img, med_img, short_img]

            self.image_list += [[exposures_path, align_ratio_path, image_path]]



    def __getitem__(self, index):
        # Read exposure times
        exposures = np.load(self.image_list[index][0])
        floating_exposures = exposures - exposures[1]


        # Read LDR images
        ldr_images = ReadImages2(self.image_list[index][2][1:])
        # print(self.image_list[index][0])
        # print(self.image_list[index][1])
        # print(self.image_list[index][2])
        # Read HDR label (gt image)
        label = imread_uint16_png(self.image_list[index][2][0], self.image_list[index][1])

        #image info
        image_name = self.image_list[index][2][-1].split('/')[-1][:4]


        # ldr images process
        gamma = 2.24
        if random.random() < 0.3:
            gamma += (random.random() * 0.2 - 0.1)

        #Canonical EV alignment

        image_short_corrected = (((ldr_images[2]**gamma)*2.0**(-1*floating_exposures[0]))**(1/gamma))
        image_medium = ldr_images[1]
        image_long_corrected = (((ldr_images[0]**gamma)*2.0**(-1*floating_exposures[2]))**(1/gamma))


        image_short_concat = np.concatenate((ldr_images[2], image_short_corrected), 2)
        image_medium_concat = np.concatenate((ldr_images[1], image_medium), 2)
        image_long_concat = np.concatenate((ldr_images[0],image_long_corrected), 2)
        # image_short_concat = ldr_images[2]
        # image_medium_concat = ldr_images[1]
        # image_long_concat = ldr_images[0]



        # data argument

        if self.crop:
                H, W, _ = ldr_images[0].shape
                x = np.random.randint(0, H - self.crop_size[0] - 1)
                y = np.random.randint(0, W - self.crop_size[1] - 1)
                img0 = image_short_concat[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
                img1 = image_medium_concat[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
                img2 = image_long_concat[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
                label = label[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)

        else:
                img0 = image_short_concat.astype(np.float32).transpose(2, 0, 1)
                img1 = image_medium_concat.astype(np.float32).transpose(2, 0, 1)
                img2 = image_long_concat.astype(np.float32).transpose(2, 0, 1)
                label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label,'image_name':image_name}
        return sample

    def __len__(self):
        return len(self.image_list)


class NTIRE_Validation_Dataset(Dataset):

    def __init__(self, val_dir, random=False):
        self.val_dir = val_dir
        self.crop_size = 500
        self.image_list = []
        self.scenes_dir_list = os.listdir(self.val_dir)
        self.random = random

        for scene in range(len(self.scenes_dir_list)):
            x = int(self.scenes_dir_list[scene][5:9])
            while True:
                number = str(x).zfill(4) + '_'
                exposures_path = osp.join(self.val_dir, self.scenes_dir_list[scene],
                                          '{}exposures.npy'.format(number))
                align_ratio_path = osp.join(self.val_dir, self.scenes_dir_list[scene],
                                            '{}alignratio.npy'.format(number))

                gt_img = os.path.join(self.val_dir, self.scenes_dir_list[scene], '{}gt.png'.format(number))
                short_img = os.path.join(self.val_dir, self.scenes_dir_list[scene],
                                             '{}short.png'.format(number))
                med_img = os.path.join(self.val_dir, self.scenes_dir_list[scene],
                                           '{}medium.png'.format(number))
                long_img = os.path.join(self.val_dir, self.scenes_dir_list[scene],
                                            '{}long.png'.format(number))

                image_path = [gt_img, long_img, med_img, short_img]

                self.image_list += [[exposures_path, align_ratio_path, image_path]]
                if int(self.scenes_dir_list[scene][10:14]) == x:
                    break
                x = x + 1


    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = np.load(self.image_list[index][0])
        floating_exposures = exposures - exposures[1]
        image_name = self.image_list[index][2][-1].split('/')[-1][:4]


        # Read LDR images
        ldr_images = ReadImages2(self.image_list[index][2][1:])
        # Read HDR label
        label = imread_uint16_png(self.image_list[index][2][0], self.image_list[index][1])

        # ldr images process
        gamma = 2.24
        if random.random() < 0.3:
            gamma += (random.random() * 0.2 - 0.1)

        # Canonical EV alignment

        image_short_corrected = (((ldr_images[2] ** gamma) * 2.0 ** (-1 * floating_exposures[0])) ** (1 / gamma))
        image_medium = ldr_images[1]
        image_long_corrected = (((ldr_images[0] ** gamma) * 2.0 ** (-1 * floating_exposures[2])) ** (1 / gamma))


        image_short_concat = np.concatenate((ldr_images[2], image_short_corrected ), 2)
        image_medium_concat = np.concatenate((ldr_images[1], image_medium), 2)
        image_long_concat = np.concatenate((ldr_images[0], image_long_corrected), 2)


        x=0
        y=0

        if self.random:
            H, W, _ = ldr_images[0].shape
            x = np.random.randint(0, H - self.crop_size - 1)
            y = np.random.randint(0, W - self.crop_size - 1)


        img0 = image_short_concat[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium_concat[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
        img2 = image_long_concat[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
        label = label[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label, 'image_name': image_name}
        return sample

    def __len__(self):
        return len(self.image_list)




class NTIRE_Test_Dataset(Dataset):

    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.image_list = []


        for i in range(60):
            x = int(i)
            number = str(x).zfill(4) + '_'
            exposures_path = osp.join(self.test_dir,'{}exposures.npy'.format(number))
            short_img = os.path.join(self.test_dir,'{}short.png'.format(number))
            med_img = os.path.join(self.test_dir, '{}medium.png'.format(number))
            long_img = os.path.join(self.test_dir,'{}long.png'.format(number))

            image_path = [long_img, med_img, short_img]

            self.image_list += [[exposures_path,image_path]]



    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = np.load(self.image_list[index][0])
        floating_exposures = exposures - exposures[1]
        image_name = self.image_list[index][1][-1].split('/')[-1][:4]
        # Read LDR images
        ldr_images = ReadImages2(self.image_list[index][1][:])

        # ldr images process
        image_short = ev_correct(ldr_images[2], floating_exposures[0], 2.24)
        image_medium = ldr_images[1]
        image_long = ev_correct(ldr_images[0], floating_exposures[2], 2.24)

        image_short_concat = np.concatenate((ldr_images[2], image_short), 2)
        image_medium_concat = np.concatenate((ldr_images[1], image_medium), 2)
        image_long_concat = np.concatenate((ldr_images[0], image_long), 2)

        img0 = image_short_concat.astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium_concat.astype(np.float32).transpose(2, 0, 1)
        img2 = image_long_concat.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)

        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'image_name': image_name}
        return sample

    def __len__(self):
        return len(self.image_list)




def custom_imshow(img):
    img = img.numpy()
    img = img.squeeze()
    print(img.shape)
    img=np.transpose(img, (1, 2, 0))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    plt.imshow(img)
    plt.show()


def process(data_loader):
    for batch_idx,sample in enumerate(data_loader):
        if(batch_idx==4):
            break
        #custom_imshow(sample['input1'])
        img=sample['input1']
        img2=sample['label']
        print(img)
        print("---------")
        print(img2)
        img = img.numpy()
        img2 = img2.numpy()
        #print(img.shape)
        img = img.squeeze()
        img2 = img2.squeeze()
        print(img.shape)
        img = np.transpose(img, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        imwrite_uint16_png('./' + "h{:04d}.png".format(batch_idx), img,'./' + "{:04d}_alignratio.npy".format(batch_idx))
        imwrite_uint16_png('./' + "h{:04d}_gt.png".format(batch_idx), img2, './' + "{:04d}_alignratio.npy".format(batch_idx))
        #print(img)
        print(sample['image_name'])


if __name__ == '__main__':
    train_dataset = NTIRE_Training_Dataset(root_dir='/home/cvip/PycharmProjects/HDR_Project/data',crop=False,crop_size=(256,256))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_dataset = NTIRE_Validation_Dataset(root_dir='/home/cvip/PycharmProjects/HDR_Project/data')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print('hello')
    process(train_loader)

