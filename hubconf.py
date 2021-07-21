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