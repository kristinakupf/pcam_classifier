import random
import torchvision.transforms as t

random.seed(1111)


class ApplyRotation(object):
    def __init__(self, num_classes, images_in, transform_val):
        self.images = images_in

        self.transform_val = ([0,90,180,270])

    def apply_transform(self):
        K = random.randrange(len(self.transform_val))
        img = t.functional.rotate(self.images, self.transform_val[K])


        return img, K


