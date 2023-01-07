import torch
import torch.backends.cudnn as cudnn

from nets.facenet import Facenet
from utils.dataloader import WebFaceDataset
from utils.utils_metrics import test

if __name__ == "__main__":
    #--------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------------#
    cuda            = True
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #--------------------------------------#
    backbone        = "mobilenet"
    #--------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    #--------------------------------------------------------#
    input_shape     = [16, 16, 3]
    #--------------------------------------#
    #   训练好的权值文件
    #--------------------------------------#
    model_path      = "webface_112x112.pth"
    #--------------------------------------#
    #   WebFace评估数据集的文件路径
    #   以及对应的txt文件
    #--------------------------------------#
    webface_dir_path    = "webface_data/"

    webface_pairs_path  = "webface_cls_test.txt"
    #--------------------------------------#
    #   评估的批次大小和记录间隔
    #--------------------------------------#
    batch_size      = 4
    log_interval    = 1
    #--------------------------------------#
    #   ROC图的保存路径
    #--------------------------------------#
    png_save_path   = "/face_mask/webface_roc_test_72x72.png"

    test_loader = torch.utils.data.DataLoader(
        WebFaceDataset(dir=webface_dir_path, pairs_path=webface_pairs_path, image_size=input_shape), batch_size=batch_size, shuffle=False)
    print(len(test_loader))

    model = Facenet(backbone=backbone, mode="predict")

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model  = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    test(test_loader, model, png_save_path, log_interval, batch_size, cuda)

