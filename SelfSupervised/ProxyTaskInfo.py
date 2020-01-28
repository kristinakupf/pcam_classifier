from SelfSupervised import Rotation

def supervised(dataset):
    if dataset=='pcam':
        num_classes=2
    if dataset=="BACH":
        num_classes=4
    pretrain_path = None
    return num_classes, pretrain_path

def rotation(dataset):
    num_classes=4
    pretrain_path = ''
    return num_classes, pretrain_path



