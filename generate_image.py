import torch
from parti.parti import Parti
import cv2
import numpy as np
from config import get_config
import argparse


def restore(x):
    x = (x + 1) * 0.5 
    # covnert to numpy
    x = x.permute(1,2,0).detach().cpu().numpy()
   
    # x = x.permute(1,2,0).detach().cpu().numpy()
    x = (255*x).astype(np.uint8)
    # x = Image.fromarray(x)
    return x


# # text = ['A woman wearing a net on her head cutting a cake.']
# ['a toilet bowl and a trash can in a bathroom']
# ['a toilet bowl and a trash can in a bathroom']
# ['The woman in the kitchen is holding a huge pan.']
# ['The woman in the kitchen is holding a huge pan.']
# ['bike riders passing Burger King in city street']
# ['bike riders passing Burger King in city street']
# ['a young boy barefoot holding an umbrella touching the horn of a cow']
# ['a young boy barefoot holding an umbrella touching the horn of a cow']
# ['A blur motorcycle against a red brick wall.']
# ['A blur motorcycle against a red brick wall.']
# ['A bathroom with multicolored tile, bathtub and pedestal sink.']
# ['A bathroom with multicolored tile, bathtub and pedestal sink.']
# ['a big bathroom that has some sinks in it']
# ['a big bathroom that has some sinks in it']
# ['some brown cabinets a black oven a tea kettle and a microwave']
# ['some brown cabinets a black oven a tea kettle and a microwave']
# ['A bathroom contains a toilet and a sink.']
# ['A bathroom contains a toilet and a sink.']
# ['There are two sinks next to two mirrors.']
# ['There are two sinks next to two mirrors.']
# ['People walking pass a horse drawn carriage sitting at the curb']
# ['People walking pass a horse drawn carriage sitting at the curb']
# ['A cop standing next to a police bike next to a man sitting on a  curb.']
# ['A cop standing next to a police bike next to a man sitting on a  curb.']
# ['Blue and orange stone clock tower with a small clock.']
# ['Blue and orange stone clock tower with a small clock.']
# ['A bathroom area with tub, sink and standup shower.']
# ['A bathroom area with tub, sink and standup shower.']
# ['A donut on the antenna of a car.']
# ['A donut on the antenna of a car.']
# ['A young lady riding a skateboard across a street.']
# ['A young lady riding a skateboard across a street.']
# ['A small child wearing headphones plays on the computer.']
# ['A small child wearing headphones plays on the computer.']
# ['A toilet, sink and mirror in the bathroom']
# ['A toilet, sink and mirror in the bathroom']
# ['A claw foot tub is in a large bathroom near a pedestal sink.']
# ['A claw foot tub is in a large bathroom near a pedestal sink.']
# ['A silver bus that is parked in a lot.']
# ['A silver bus that is parked in a lot.']
# ['A section of traffic coming to a stop at an intersection.']
# ['A section of traffic coming to a stop at an intersection.']
# ['A woman cutting a large white sheet cake.']
# ['A woman cutting a large white sheet cake.']
# ['A young boy stares up at the computer monitor.']
# ['A young boy stares up at the computer monitor.']
# ['A shower stall with interesting tile is the focal point.']
# ['A shower stall with interesting tile is the focal point.']
# ['A parking meter on a street with cars']
# ['A parking meter on a street with cars']
# ['Two men that are standing in a kitchen.']
# ['Two men that are standing in a kitchen.']
# ['A clean, spacious bathroom with a large shower stall.']
# ['A clean, spacious bathroom with a large shower stall.']
# ['a bathroom with just a toliet and a sink in it']
# ['a bathroom with just a toliet and a sink in it']
# ['A kitchen with wood floors and lots of furniture.']
# ['A kitchen with wood floors and lots of furniture.']
# ['A white toilet sitting in a bathroom next to a shower.']
# ['A white toilet sitting in a bathroom next to a shower.']
# ['A woman on a motor cycle on a city street']
# ['A woman on a motor cycle on a city street']
# ['Two bikers, one in front of a building, the other in the city.']
# ['Two bikers, one in front of a building, the other in the city.']
# ['a glass walled shower in a home bathroom']
# ['a glass walled shower in a home bathroom']
# ['Horses grazing in a field by a large home.']
# ['Horses grazing in a field by a large home.']
# ['A bathroom with a white toilet in the middle of the wall and a sun wall decor above it.']
# ['A bathroom with a white toilet in the middle of the wall and a sun wall decor above it.']
# ['white vanity that opens up to a bathroom with shower']
# ['white vanity that opens up to a bathroom with shower']
# ['A large boat full of men is sitting on a cart ']
# ['A large boat full of men is sitting on a cart ']
# ['There is a potted plant on the back of a toilet']
# ['There is a potted plant on the back of a toilet']
# ['A kitchen is shown with a variety of items on the counters.']
# ['A kitchen is shown with a variety of items on the counters.']
# ['A woman eating fresh vegetables from a bowl.']
text = ['A woman eating fresh vegetables from a bowl.']
# ['A baby wearing gloves, lying next to a teddy bear']
# ['A baby wearing gloves, lying next to a teddy bear']
# ['A white and beige tiled bathroom and adjoining walk-in closet.']
# ['A white and beige tiled bathroom and adjoining walk-in closet.']
# ['A blurry picture of a cat standing on a toilet.']
# ['A blurry picture of a cat standing on a toilet.']
# ['A toilet with a sink and the door opened']
# ['A toilet with a sink and the door opened']
# ['A dark and cluttered storage area with wood walls.']
# ['A dark and cluttered storage area with wood walls.']
# ['A large clock tower towering over a small city.']
# ['A large clock tower towering over a small city.']
# ['horses eating grass in a field with trees in the background']
# ['horses eating grass in a field with trees in the background']
# ['A commercial dish washing station with a toilet in it.']
# ['A commercial dish washing station with a toilet in it.']
# ['Interior shot of bathroom in the process of remodeling.']
# ['Interior shot of bathroom in the process of remodeling.']
# ['A bathroom sink with a mirror and medicine cabinet.']



# text = ['Man riding a motor bike on a dirt road on the countryside.']
# text = ['A woman wearing a net on her head cutting a cake.']
# text = ["a cat and a dog sitting on a couch"]
# text = ['A blur motorcycle against a red brick wall.']

# text = ['Children sitting at computer stations on a long table.']
# text = ['a cat eating a piece of bread on a table']


parser = argparse.ArgumentParser(description='CLIFF')
parser.add_argument('--cfg', type=str, help='config file path')
opt = parser.parse_args()
cfg = get_config(opt)


model = Parti(cfg).cuda()


# load model
model.load_state_dict(torch.load("checkpoints/checkpoint_iter160000_exp1.pth")['state_dict'])
model.eval()


# generate image
with torch.no_grad():
    img = model.generate(text)[0]
    img = restore(img)
    # cv2.imwrite("result.png", img)
    cv2.imshow("image", img)
    cv2.waitKey(0)

