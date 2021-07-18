#Hub config file
dependencies = ['torch', 'torchvision']
from torchvision.models import squeezenet as squeezenet_model

def squeezenet(pretrained=False, **kwargs):
	"The statements we write here is displayed when \
	we call hub.help() on this repository"
	model = squeezenet_model(pretrained=pretrained, **kwargs)
	return model
	
