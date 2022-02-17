import sys
import os
import shutil
import torch
import time
import cv2.cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5 import uic, QtCore, QtGui, QtWidgets
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm
from network import archs
from utils.dataset import ISBI_Loader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='dsb2018_96_NestedUNet_woDS', help='model name')

    args = parser.parse_args()

    return args

class ImageProcess_ui:
    def __init__(self):
        self.ui = uic.loadUi("ImageProcess_ui.ui")
        self.ui.pushButton_select.clicked.connect(self.openimage)
        self.ui.pushButton_sementic.clicked.connect(self.grayimage)

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self.ui, "打开图片", "", "All Files(*);;*.png;;*.jpg")
        jpg = QtGui.QPixmap(imgName).scaled(self.ui.label_original_show.width(), self.ui.label_original_show.height())
        self.ui.label_original_show.setPixmap(jpg)
        self.ui.image_initial = imgName
        print(imgName)
        img_path = imgName
        target_path = 'tempt_dir/01/tempt.png'
        os.makedirs('tempt_dir', exist_ok=True)
        os.makedirs('tempt_dir/01', exist_ok=True)
        shutil.copyfile(img_path, target_path)

    def grayimage(self):
        args = parse_args()
        with open('Result/%s/config.yml' % args.name, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        cudnn.benchmark = True

        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(localtime, "：图片处理中...")
        model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'],
                                               config['deep_supervision'])

        model = model.cuda()
        model.load_state_dict(torch.load('Result/%s/model.pth' % config['name']))
        model.eval()

        image_pathh = './tempt_dir/*'
        label_pathh = './tempt_dir/*'
        val_dataset = ISBI_Loader(image_pathh, label_pathh)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False)

        with torch.no_grad():
            for input, target in tqdm(val_loader, total=len(val_loader)):
                input = input.cuda()

                if config['deep_supervision']:
                    output = model(input)[-1]
                else:
                    output = model(input)

                output = torch.sigmoid(output).cpu().numpy()

                for i in range(len(output)):
                    for c in range(config['num_classes']):
                        cv.imwrite('pred.png', (output[i, c] * 255).astype('uint8'))

        pix = QPixmap('pred.png')
        self.ui.label_sementic_show.setPixmap(pix)
        self.ui.label_sementic_show.setScaledContents(True)

        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(localtime, "：图片处理完成")
        shutil.rmtree('tempt_dir')
        os.remove('./pred.png')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    imageProcess_ui = ImageProcess_ui()
    imageProcess_ui.ui.show()
    app.exec_()