import torch
from torchviz import make_dot
import torch.nn as nn
from model import *
from utils import *

def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)

model =  DiscriminatorGAN()#(3,500, features = [32, 64, 128, 256], act_fxn = nn.ReLU(True))
# model = Encoder()
x = torch.randn((1, 3, 224, 224))

# # Generate a hiddenlayer summary
# summary = hl.build_graph(model, x)

# # Display the summary
# summary.display()

# # Save the visualization
# summary.save('viz/'+str(Model._get_name()), format="png")
y = model(x,x)
# for param in model.parameters():
#     param.requires_grad = False
params = dict(model.named_parameters())
# print(params.keys())

dot = torchviz.make_dot(y, params=params)
resize_graph(dot)
dot.render('viz/'+str(model._get_name()), format='png')

