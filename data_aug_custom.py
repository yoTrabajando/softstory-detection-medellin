import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image
import random
from torch import randn_like

# funcion que rota la imagen y rellena los espacios vacios con un reflejo de la misma imagen
class RandomRotationReflect:
    def __init__(self, degrees=30, pad_pct=0.25, interp=T.InterpolationMode.BILINEAR):
        self.degrees = degrees
        self.pad_pct = pad_pct
        self.interp = interp

    def __call__(self, img):
        # get H, W for PIL or Tensor
        if isinstance(img, Image.Image):
            w, h = img.size
        else:  # Tensor [C,H,W]
            h, w = img.shape[-2], img.shape[-1]

        # reflect-pad first (so rotation samples inside the padded area)
        pad = int(max(h, w) * self.pad_pct)
        img = T.Pad(pad, padding_mode="reflect")(img)

        # rotate (fill value won't matter because we padded)
        angle = random.uniform(-self.degrees, self.degrees)
        img = F.rotate(img, angle=angle, interpolation=self.interp, expand=False, fill=0)

        # crop back to original size
        img = T.CenterCrop((h, w))(img)
        return img
    
def rotation_reflect(x, angle, pad_pct = 0.25):
    if isinstance(x, Image.Image):
        w, h = x.size
    else:  # Tensor [C,H,W]
        h, w = x.shape[-2], x.shape[-1]

    pad = int(max(h, w) * pad_pct)
    img = T.Pad(pad, padding_mode="reflect")(x)
    img = F.rotate(img, angle=angle, interpolation=T.InterpolationMode.BILINEAR, expand=False, fill=0)

    img = T.CenterCrop((h, w))(img)
    return img

def zoom_image(x, ratio):
    w, h = x.size
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    return F.center_crop(x, [new_h, new_w])

# Añade a la imagen ruido con una distribución normal  
def gaussian_noise(x, mean = 0, std = 1):
    return x + randn_like(x) * std + mean