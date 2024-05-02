import numpy as np
from PIL import Image
# from imgutils import myimshows
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import torch
from zoedepth.utils.misc import colorize

# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

image = Image.open(r"000000.png").convert("RGB")  # load
#image = Image.open(r"D:\AI_toolV.13\TouKui_jc/part2_000509.jpg").convert("RGB")  # load
#depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image
depth = zoe.infer_pil(image)
colored = colorize(depth)
# save colored output
fpath_colored = "output_colored.png"
colored_img=Image.fromarray(colored)
colored_img.save(fpath_colored)

# myimshows([np.array(image),colored],["ori","depth"])
