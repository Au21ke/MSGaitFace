from PIL import Image
import numpy as np
from .facenet import Facenet
import os


def Face_Dist(ratio=0.4,facemask=True):
    model = Facenet()
    probe_x = []
    gallery_x = []
    test_pth = ""

    for id in os.listdir(test_pth):
        n = 0
        for pic in os.listdir(os.path.join(test_pth, id)):
            img = Image.open(os.path.join(test_pth, id, pic))
            #if facemask:
            #    img = Face_Mask(img, ratio)
            if n < 7:
                gallery_x.append(img)
            elif n == 7:
                probe_x.append(img)
            else:
                break
            n = n + 1
    # print(len(gallery_x)) [450]
    # print(len(probe_x))    [50]
    dist = np.zeros((len(probe_x), len(gallery_x)))
    # print(dist.shape)
    for i in range(len(probe_x)):
        for j in range(len(gallery_x)):
            dist[i, j] = model.detect_image(probe_x[i], gallery_x[j])  # [50,450]
    # 求均值
    face_dist = np.zeros((len(probe_x), len(probe_x)))  # [50,50]
    
    for i in range(len(probe_x)):
        c = 0
        for j in range(len(probe_x)):
            face_dist[i, j] = np.min(dist[i, c:c + 7])
            c = c + 7

    return face_dist

def Face_Mask(image, ratio):
    # print(image.shape[0])
    image = np.asarray(image).copy()
    w, h = image.shape[0], image.shape[1]
    image[-int(w * ratio):, :] = 0
    image = Image.fromarray(image)
    #import imageio
    #imageio.imsave('test.png', image)
    return image

if __name__ == "__main__":
    print(Face_Dist())


