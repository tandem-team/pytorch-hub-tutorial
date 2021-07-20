#Hub config file
dependencies = ['torch', 'torchvision']
import torch
from torchvision.models import squeezenet1_0 as _squeezenet_model

def squeezenet(pretrained=False, **kwargs):
	"""The statements we write here is displayed when \
	we call hub.help() on this repository"""
	model = _squeezenet_model(pretrained=pretrained, **kwargs)
	
	checkpoint = 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth'
	model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
	return model

squeezenet()