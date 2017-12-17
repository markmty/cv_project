import torch
from torchvision import transforms, datasets
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

CONTENT_IMAGE_PATH = '/Users/Jade/Downloads/a.jpg'
MODEL_PATH = '/Users/Jade/NYU/cv/cv_project/epoch_0.model'
OUTPUT_IMAGE_PATH = '/Users/Jade/NYU/cv/cv_project/test1.jpg'

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

def bgr2rgb(image):
	(b, g, r) = torch.chunk(image, 3)
	image = torch.cat((r, g, b))
	return image

def savergb(image,filename):
	if is_cuda:
		img = image.clone().cpu().clamp(0, 255).numpy()
	else:
		img = image.clone().clamp(0,255).numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	img = Image.fromarray(img)
	img.save(filename)

content_image = load_image(CONTENT_IMAGE_PATH,IMAGE_SIZE)
content_image = content_image.unsqueeze(0)

if is_cuda:
	content_image.cuda()

content_image = rgb2bgr(content_image)
content_image = Variable(content_image, volatile=True)

style_model = transformer_net()
style_model.load_state_dict(torch.load(MODEL_PATH))

if is_cuda:
	style_model.cuda()

output = style_model(content_image)
result = bgr2rgb(output.data[0])
savergb(result,OUTPUT_IMAGE_PATH)