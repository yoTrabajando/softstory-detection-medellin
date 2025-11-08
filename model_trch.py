# Librerias
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class model_highlight(nn.Module):
    # Se inicializan las variables del modelo 
        # pretrained: modelo preentrenado
        # out_pretrained: cantidad de salidas del modelo preentrenado
        # hidden_layers: cantidad de neuronas por capa oculta, siendo cada capa oculta un elemento de un arreglo 1xn
        # output: neuronas de salida (en este caso 1 por la naturaleza de los problemas de clasificación binaria)
    def __init__(self, channels_in, out_conv, hidden_layers, probability, output = 1):

        # se juntan las salidas del modelo, las capas ocultas y la salida final
        layers = np.append(np.array([out_conv]), np.array([hidden_layers]))
        layers = np.append(layers, np.array([output]))

        # se heradan las propiedades del modulo nn.Module
        super(model_highlight, self).__init__()

        self.out_conv = out_conv

        # se inicializan las salidas del modelo preentrenado y el modelo preentrenaod para ser usados luego dentro de la clase
        #5x256x256
        
        self.conv1 = nn.Conv2d(
            in_channels = channels_in,
            out_channels = 8,
            kernel_size = 3,
            stride = 1,
            padding = 0
        ) #16x254x254

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)

        self.bn1 = nn.BatchNorm2d(num_features = 8)

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 3,
            stride = 1,
            padding = 0
        ) #16x252x252

        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)

        self.bn2 = nn.BatchNorm2d(num_features = 16)

        self.drop1 = nn.Dropout2d(probability)

        self.pool1 = nn.MaxPool2d(2, 2) #16x126x126

        self.conv3 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 0
        ) #32x124x124

        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv3.bias, 0)

        self.bn3 = nn.BatchNorm2d(num_features = 32)

        self.conv4 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            stride = 1,
            padding = 0
        ) #64x122x122

        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv4.bias, 0)

        self.bn4 = nn.BatchNorm2d(num_features = 64)

        self.drop2 = nn.Dropout2d(probability)

        self.pool2 = nn.MaxPool2d(2, 2) #64x61x61

        self.conv5 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            stride = 1,
            padding = 0
        ) #128x59x59

        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv5.bias, 0)

        self.bn5 = nn.BatchNorm2d(num_features = 128)

        self.conv6 = nn.Conv2d(
            in_channels = 128,
            out_channels = 256,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ) #256x58x58

        nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv6.bias, 0)

        self.bn6 = nn.BatchNorm2d(num_features = 256)

        self.drop3 = nn.Dropout2d(probability)

        self.pool3 = nn.MaxPool2d(2, 2) #256x29x29

        self.conv7 = nn.Conv2d(
            in_channels = 256,
            out_channels = 512,
            kernel_size = 3,
            stride = 1,
            padding = 0
        ) #512x27x27

        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv5.bias, 0)

        self.bn7 = nn.BatchNorm2d(num_features = 512)

        self.conv8 = nn.Conv2d(
            in_channels = 512,
            out_channels = 1024,
            kernel_size = 3,
            stride = 1,
            padding = 0
        ) #1024x25x25

        nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv6.bias, 0)

        self.bn8 = nn.BatchNorm2d(num_features = 1024)

        self.drop4 = nn.Dropout2d(probability)

        self.pool4 = nn.MaxPool2d(5, 5) #1024x5x5



        # se crean 2 listas de capas distintas: una para capas ocultas y otra para capas de batch normalization
        self.hidden = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        # se establece el limite de capas para el forward method
        self.limit_batch = len(layers)

        # se crean las capas ocultas y de batch normalization siendo que las capas batch norm siempre van entre 2 capas ocultas
        for i, input_size, output_size in zip(range(self.limit_batch), layers, layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))
            if i < self.limit_batch-2:
                self.batchnorm.append(nn.BatchNorm1d(output_size))
    
    # se define el forward method
    def forward(self, x):
        x = F.sigmoid(self.bn1(self.conv1(x)))
        
        x = F.sigmoid(self.bn2(self.conv2(x)))
        
        x = self.pool1(self.drop1(x))
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.pool2(self.drop2(x))
        
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = F.relu(self.bn6(self.conv6(x)))
        
        x = self.pool3(self.drop3(x))
        
        x = F.relu(self.bn7(self.conv7(x)))
        
        x = F.relu(self.bn8(self.conv8(x)))

        x = self.pool4(self.drop4(x))

        x = torch.flatten(x, 1)
        # se crea un for loop donde las salidas del modelo preentrenado hacen el siguiente ciclo:
            # 1) pasan por una capa de perceptrones
            # 2) pasan por una capa batch norm
            # 3) pasan por una función ReLU
        # justo antes de finalizar el loop, se pasa por ultima vez por una capa de una neurona para que sea pasado por una función sigmoide
        for i, dense in zip(range(self.limit_batch), self.hidden):
            x = dense(x)
            if i < self.limit_batch-2:
                x = F.relu(self.batchnorm[i](x))
            else:
                x = F.sigmoid(x)
        return x