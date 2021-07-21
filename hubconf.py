#Hub config file

import torch
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1

def squeezenet(pretrained_t=False):
	"The statements we write here is displayed when \
	we call hub.help() on this repository"
	model = squeezenet1_0(pretrained=pretrained_t)
	
	checkpoint = 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth'
	model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
	return model
	
#Import packages
import torch
import torchvision
from PIL import Image
import requests

def image_to_tensor(url_t):
	#Basic transformations for Imagenet Object detection
	from torchvision import transforms
	transform = transforms.Compose([            #[1]
	transforms.Resize(256),                    #[2]
	transforms.CenterCrop(224),                #[3]
	transforms.ToTensor(),                     #[4]
	transforms.Normalize(                      #[5]
	mean=[0.485, 0.456, 0.406],                #[6]
	std=[0.229, 0.224, 0.225]                  #[7]
	)])

	#Open image, transform it and add dimension via `unsqueeze(0)`
	im = Image.open(requests.get(url_t[1], stream=True).raw)
	img_t = transform(im)
	return torch.unsqueeze(img_t, 0)

def squeezenet_tensor_out_util(url_list):

	
	out_tensor = image_to_tensor(url[0])
	for url in url_list[1:]:
		torch.cat((out_tensor, image_to_tensor(url)), 0)
		
	return out_tensor