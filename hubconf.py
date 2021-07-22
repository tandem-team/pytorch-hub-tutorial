#Hub config file
import torch
import requests
from PIL import Image
from torchvision import transforms
from torchvision.models.squeezenet import squeezenet1_0


#Entry-point function definition which serves the model definition(and weights, if specified)

def squeezenet(pretrained_t=False):
	"The statements we write here is displayed when \
	we call hub.help() on this repository"
	model = squeezenet1_0(pretrained=pretrained_t)
	
	checkpoint = 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth'
	model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
	return model
	
#Utility function returns tensor from the given url parameter
def image_to_tensor(url_t):
	#Basic transformations for Imagenet Object detection
	transform = transforms.Compose([            #[1]
	transforms.Resize(256),                    #[2]
	transforms.CenterCrop(224),                #[3]
	transforms.ToTensor(),                     #[4]
	transforms.Normalize(                      #[5]
	mean=[0.485, 0.456, 0.406],                #[6]
	std=[0.229, 0.224, 0.225]                  #[7]
	)])

	#Open image, transform it and add dimension via `unsqueeze(0)` and return
	im = Image.open(requests.get(url_t, stream=True).raw)
	img_t = transform(im)
	return torch.unsqueeze(img_t, 0)


def squeezenet_tensor_out_util(url_list):
	#Input -> url list
	#Ouput -> tensor list of images
	if(len(url_list)<1):
		return 0
	
	out_tensor = image_to_tensor(url_list[0])
	#If only 1 url then return
	if(len(url_list)==1):
		return out_tensor
	#If more than 1, then append and form batch of images	
	for url in url_list[1:]:
		out_tensor = torch.cat((out_tensor, image_to_tensor(url)), 0)
		
	return out_tensor
