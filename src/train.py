import torch
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from vgg16 import vgg16
from transformer_net import transformer_net
from PIL import Image
import numpy as np

# check if gpu is available
is_cuda = torch.cuda.is_available()

# parameters
IMAGE_SIZE=256
BATCH_SIZE=4
NUM_WORKERS=4
LEARNING_RATE=0.001
EPOCHS=1
CONTENT_WEIGHT=1.0
STYLE_WEIGHT=5.0
SEED=1

# file path
DATASET_PATH='/scratch/tm2749/val_data'
STYLE_IMAGE_PATH='/scratch/tm2749/style_image/mosaic.jpg'
MODEL_PATH='/scratch/tm2749/saved_model'

if is_cuda:
    torch.cuda.manual_seed(SEED)
else:
    torch.manual_seed(SEED)

# data transforms
transforms = transforms.Compose([
    transforms.Scale(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))])

# load training dataset
train_dataset = datasets.ImageFolder(DATASET_PATH, transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# init vgg16
vgg16 = vgg16()
vgg16.load_state_dict(torch.load(MODEL_PATH+'/vgg16.weight'))
if is_cuda:
    vgg16.cuda()

# init transformer net
trans_net = transformer_net()
if is_cuda:
    trans_net.cuda()

optimizer = optim.Adam(trans_net.parameters(), LEARNING_RATE)
mse_loss = torch.nn.MSELoss()

# load style image
def load_image(file, size):
    img = Image.open(file)
    img = img.resize((size, size), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def rgb2bgr(image):
    image = image.transpose(0, 1)
    (r, g, b) = torch.chunk(image, 3)
    image = torch.cat((b, g, r))
    image = image.transpose(0, 1)
    return image

def subtract_mean(X):
    tensor_type = type(X.data)
    mean = tensor_type(X.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    X = X.sub(Variable(mean))
    return X

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w*h)
    features_t = features.transpose(1,2)
    gram = features.bmm(features_t) /(ch*h*w)
    return gram

style_image = load_image(STYLE_IMAGE_PATH, IMAGE_SIZE)
style_image = style_image.repeat(BATCH_SIZE, 1, 1, 1)
style_image = rgb2bgr(style_image)
if is_cuda:
    style_image = style_image.cuda()
style_x = Variable(style_image, volatile=True)
style_x = subtract_mean(style_x)

features_style = vgg16(style_x)
gram_style = [gram_matrix(y) for y in features_style]
trans_net.train()
for i in range(EPOCHS):
    #count = 0
    for batch_id, batch in enumerate(train_loader):
        optimizer.zero_grad()
        current_bs = len(batch[1])
        data = batch[0].clone()
        data = rgb2bgr(data)

        if is_cuda:
            data = data.cuda()

        x = Variable(data.clone())
        y = trans_net(x)
        y = subtract_mean(y)
        features_y = vgg16(y)

        x_content = Variable(data.clone(), volatile=True)
        x_content = subtract_mean(x_content)
        features_x_content = vgg16(x_content)

        feature_relu = Variable(features_x_content[1].data, requires_grad=False)
        content_loss = CONTENT_WEIGHT * mse_loss(features_y[1], feature_relu)

        style_loss = 0.0
        for j in range(len(features_y)):
            gram1 = Variable(gram_style[j].data, requires_grad=False)
            gram2 = gram_matrix(features_y[j])
            style_loss += STYLE_WEIGHT * mse_loss(gram2, gram1[0:current_bs, :, :])

        loss = content_loss + style_loss
        loss.backward()
        optimizer.step()

    trans_net.eval()
    trans_net.cpu()
    model_name = "epoch_"+str(i)+".model"
    torch.save(trans_net.state_dict(), MODEL_PATH+"/"+model_name)

    print("saved "+model_name)












