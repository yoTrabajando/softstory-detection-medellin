# Librerias
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpigmg
import numpy as np

# Librerias de aumento de datos propias
from data_aug_custom import rotation_reflect, zoom_image, gaussian_noise

# Función de aumento de datos que si es verdadera, aplica funciones como: random flip o random rotation
# si no es verdadera no aplica funciones de aumento de datos
def augmentation(x, mean, std, rand_vec = [], add_noise = False, size = 256):
    if len(rand_vec) > 0:
        if rand_vec[0] >= 0.5:
            x = F.hflip(x)

        angle = rand_vec[1]*20-10
        x = rotation_reflect(x, angle = angle)

        ratio = rand_vec[2]*0.2+0.8
        x = zoom_image(x, ratio = ratio)

        x = F.resize(x, [size, size])
        x = F.to_tensor(x)

        x = F.normalize(x, 
                        mean = mean,
                        std = std)

        if add_noise:
            x = gaussian_noise(x, mean = 0.0, std = 0.01)

    else:
        x = F.resize(x, [size, size])
        x = F.to_tensor(x)

        x = F.normalize(x, 
                        mean = mean, 
                        std = std)
    return x


class images_ds(Dataset):
    # Inicializa todas las variables
        # dataset: pandas dataframe que tiene los atributos de las imágenes
        # file_path_col: columna que tiene los caminos de los archivos de las imágenes
        # labels_col: columna que posee los supuestos
        # data_transform: función para aumento de datos
    def __init__(self, dataset, file_path_col, highlight_col, labels_col, data_transform, data_augmentation, mean_img = [0., 0., 0.], std_img = [1., 1., 1.], mean_high = [0., 0., 0.], std_high = [1., 1., 1.]):
        self.file_path_col = file_path_col
        self.highlight_col = highlight_col
        self.labels_col = labels_col
        self.dataset = dataset

        self.file_path = dataset[file_path_col]
        self.labels = dataset[labels_col]
        self.highlight = dataset[highlight_col]

        self.transform = data_transform
        self.data_augmentation = data_augmentation
        self.mean_img = mean_img
        self.std_img = std_img
        self.mean_high = mean_high
        self.std_high = std_high


        self.len = dataset.shape[0]

    # función que dice la cantidad de elementos
    def __len__(self):
        return self.len

    # función que da la imagen como tensor incluyendo el aumento de datos con su respectivo supuesto    
    def __getitem__(self, index):
        image_path = self.file_path[index]
        highlight_path = self.highlight[index]

        if self.data_augmentation:
            rand_vec = np.random.rand(3)
        else:
            rand_vec = []

        main_image = self.transform(Image.open(image_path).convert("RGB"), 
                                    rand_vec = rand_vec, 
                                    add_noise = True,
                                    mean = self.mean_img,
                                    std = self.std_img)
        
        highlight_image = self.transform(Image.open(highlight_path).convert("RGB"),
                                         rand_vec = rand_vec,
                                         mean = self.mean_high,
                                         std = self.std_high)
        
        label = torch.tensor(self.labels[index], dtype = torch.float32).reshape(-1)
        return torch.cat((main_image, highlight_image[0:2]), dim = 0), label
    
class sematic_segmentation_ds(Dataset):
    # Inicializa todas las variables
        # dataset: pandas dataframe que tiene los atributos de las imágenes
        # file_path_col: columna que tiene los caminos de los archivos de las imágenes
        # labels_col: columna que posee los supuestos
        # data_transform: función para aumento de datos
    def __init__(self, dataset, file_path_col, highlight_col, data_transform, data_augmentation, mean_img = [0., 0., 0.], std_img = [1., 1., 1.]):
        self.file_path_col = file_path_col
        self.highlight_col = highlight_col
        self.dataset = dataset

        self.file_path = dataset[file_path_col]
        self.highlight = dataset[highlight_col]

        self.transform = data_transform
        self.data_augmentation = data_augmentation
        self.mean_img = mean_img
        self.std_img = std_img


        self.len = dataset.shape[0]

    # función que dice la cantidad de elementos
    def __len__(self):
        return self.len

    # función que da la imagen como tensor incluyendo el aumento de datos con su respectivo supuesto    
    def __getitem__(self, index):
        image_path = self.file_path[index]
        highlight_path = self.highlight[index]

        if self.data_augmentation:
            rand_vec = np.random.rand(3)
        else:
            rand_vec = []

        main_image = self.transform(Image.open(image_path).convert("RGB"), 
                                    rand_vec = rand_vec, 
                                    add_noise = True,
                                    mean = self.mean_img,
                                    std = self.std_img)
        
        
        highlight_image = self.transform(Image.open(highlight_path).convert("RGB"),
                                         rand_vec = rand_vec,
                                         mean = [0., 0., 0.],
                                         std = [1., 1., 1.]
                                         )[0:2]
        
        highlight_image = torch.squeeze(highlight_image[0] - highlight_image[1])
        highlight_image = torch.where(highlight_image > 0, torch.tensor(1), highlight_image)
        highlight_image = torch.where(highlight_image < 0, torch.tensor(2), highlight_image).to(torch.long)
        
        return main_image, highlight_image




# función que muestra la imagen para ver las modificaciones que hace el aumento de datos
def show_image(tensor, highlight = False, mean = [0., 0., 0.], std = [1., 1., 1.]):
    to_pil = T.ToPILImage()
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    if highlight and tensor.ndimension() == 3:
        img_tensor = torch.cat((tensor[3:], torch.zeros((1, tensor.size(1), tensor.size(2)),
                                            dtype = torch.float32,
                                            requires_grad = False)),
                        dim = 0
                        )
        img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]

    elif highlight:
        channel1_tensor = torch.unsqueeze(torch.where(tensor == 1, torch.tensor(1), torch.tensor(0)), dim = 0).to(torch.float32)
        channel2_tensor = torch.unsqueeze(torch.where(tensor == 2, torch.tensor(1), torch.tensor(0)), dim = 0).to(torch.float32)

        img_tensor = torch.cat((channel1_tensor, channel2_tensor), dim = 0)

        img_tensor = torch.cat((img_tensor, torch.zeros((1, img_tensor.size(1), img_tensor.size(2)),
                                            dtype = torch.float32,
                                            requires_grad = False)),
                        dim = 0
                        )
        
    else:
        img_tensor = tensor[0:3] * std[:, None, None] + mean[:, None, None]
    
    img = to_pil(img_tensor).convert("RGB")
    print(img)
    plt.imshow(img)
    plt.axis("off")
    plt.show()