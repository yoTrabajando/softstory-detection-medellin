# Librerias
import numpy as np
from torch import save, load, no_grad
import torch
from datetime import datetime

# clase para el entrenamiento del modelo

    # se inicializan las variables:
        # model: modelo de pytorch
        # train_loader: conjunto de datos de entrenamiento cargados por batches
        # val_loader: conjunto de datos de prueba cargados por batches
        # optimizer: optimizador utilizado
        # loss_function: función de costo
        # metric: métrica con la cual el modelo va a ser medido durante el entrenamiento

    # método que se activa cuando se invoca el objeto con las epochs que se quiera entrenar
def model_train(epochs, model, train_loader, val_loader, optimizer, loss_function, metric, save = False):
    
    acc_max = 0

    # arreglos de métricas
    loss_train_list = np.array([])
    acc_train_list = np.array([])
    loss_val_list = np.array([])
    acc_val_list = np.array([])
    # for loop para epoch
    for epoch in range(epochs):
        loss_train_sublist = []
        acc_train_sublist = []
        loss_val_sublist = []
        acc_val_sublist = []
        # for loop para entrenar con todos los batches
        for x, y in train_loader:
            # se cambia el estado del modelo a entrenamiento
            model.train()
            # se limpian los gradientes
            optimizer.zero_grad()
            # se implementa el forward method del modelo para obtener valores predichos
            yhat = model(x)
            # se obtiene el valor de costo de entrenamiento
            loss = loss_function(yhat, y)
            # se calcula el Accuracy de entrenamiento
            acc = metric(yhat, y)
            # se agregan las métricas a una lista para luego ser promediadas
            loss_train_sublist.append(loss.data.item())
            acc_train_sublist.append(acc)
            # se actualizan los graientes
            loss.backward()
            # se modifican los parámentros
            optimizer.step()
        # se promedian las métricas por epoch
        loss_mean = np.mean(loss_train_sublist)
        acc_mean = np.mean(acc_train_sublist)
        # se agregan los resultados para luego graficarlos
        loss_train_list = np.append(loss_train_list, loss_mean)
        acc_train_list = np.append(acc_train_list, acc_mean)
        # for loop para la prueba
        for x_val, y_val in val_loader:
            # se establece el modelo en modo evaluación
            model.eval()
            # se implementa el forward method del modelo para obtener valores predichos
            yhat_val = model(x_val)
            # se obtiene el valor de costo de entrenamiento
            loss_val = loss_function(yhat_val, y_val)
            # se calcula el Accuracy de entrenamiento
            acc_val = metric(yhat_val, y_val)
            # se agregan las métricas a una lista para luego ser promediadas
            loss_val_sublist.append(loss_val.data.item())
            acc_val_sublist.append(acc_val)
        
        # se promedian las métricas por epoch
        loss_mean_val = np.mean(loss_val_sublist)
        acc_mean_val = np.mean(acc_val_sublist)
        # se agregan los resultados para luego graficarlos
        loss_val_list = np.append(loss_val_list, loss_mean_val)
        acc_val_list = np.append(acc_val_list, acc_mean_val)
        # se muestra en pantalla las métricas por epoch durante el entrenamiento

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss_mean:.2f} | Train Accuracy: {acc_mean:.2f} | Validation Loss: {loss_mean_val:.2f} | Validation Accuracy: {acc_mean_val:.2f}")

        if save and acc_mean_val > acc_max:
            torch.save(model.state_dict(), "saved_models/best_model.pth")
            acc_max = acc_mean_val
            print('Model saved!')
    # Las métricas se guarda en un diccionario
    return {
        "train_loss": loss_train_list,
        "train_accuracy": acc_train_list,
        "val_loss": loss_val_list,
        "val_accuracy": acc_val_list
    }

# clase para el entrenamiento del modelo
class semseg_train():
    # se inicializan las variables:
        # model: modelo de pytorch
        # train_loader: conjunto de datos de entrenamiento cargados por batches
        # val_loader: conjunto de datos de prueba cargados por batches
        # optimizer: optimizador utilizado
        # loss_function: función de costo
        # metric: métrica con la cual el modelo va a ser medido durante el entrenamiento
    def __init__(self, model, train_loader, val_loader, optimizer, loss_function, metric):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metric = metric
        self.metrics = None
    
    # método que se activa cuando se invoca el objeto con las epochs que se quiera entrenar
    def __call__(self, epochs):
        
        # arreglos de métricas
        loss_train_list = np.array([])
        acc_train_list = np.array([])
        iou1_train_list = np.array([])
        iou2_train_list = np.array([])
        loss_val_list = np.array([])
        acc_val_list = np.array([])
        iou1_val_list = np.array([])
        iou2_val_list = np.array([])
        

        # for loop para epoch
        for epoch in range(epochs):
            loss_train_sublist = []
            acc_train_sublist = []
            iou1_train_sublist = []
            iou2_train_sublist = []
            loss_val_sublist = []
            acc_val_sublist = []
            iou1_val_sublist = []
            iou2_val_sublist = []

            # for loop para entrenar con todos los batches
            for x, y in self.train_loader:
                # se cambia el estado del modelo a entrenamiento
                self.model.train()
                # se limpian los gradientes
                self.optimizer.zero_grad()
                # se implementa el forward method del modelo para obtener valores predichos
                logits = self.model(x)
                yhat = torch.argmax(logits, dim = 1)
                # se obtiene el valor de costo de entrenamiento
                loss = self.loss_function(logits, y)
                # se calcula el Accuracy de entrenamiento
                acc, iou1, iou2 = self.metric(yhat, y)
                # se agregan las métricas a una lista para luego ser promediadas
                loss_train_sublist.append(loss.data.item())
                acc_train_sublist.append(acc)
                iou1_train_sublist.append(iou1)
                iou2_train_sublist.append(iou2)
                # se actualizan los graientes
                loss.backward()
                # se modifican los parámentros
                self.optimizer.step()

            # se promedian las métricas por epoch
            loss_mean = np.mean(loss_train_sublist)
            acc_mean = np.mean(acc_train_sublist)
            iou1_mean = np.mean(iou1_train_sublist)
            iou2_mean = np.mean(iou2_train_sublist)

            # se agregan los resultados para luego graficarlos
            loss_train_list = np.append(loss_train_list, loss_mean)
            acc_train_list = np.append(acc_train_list, acc_mean)
            iou1_train_list = np.append(iou1_train_list, iou1_mean)
            iou2_train_list = np.append(iou2_train_list, iou2_mean)

            # for loop para la prueba
            for x_val, y_val in self.val_loader:
                # se establece el modelo en modo evaluación
                self.model.eval()
                # se implementa el forward method del modelo para obtener valores predichos
                logits_val = self.model(x_val)
                yhat_val = torch.argmax(logits_val, dim = 1)
                # se obtiene el valor de costo de entrenamiento
                loss_val = self.loss_function(logits_val, y_val)
                # se calcula el Accuracy de entrenamiento
                acc_val, iou1_val, iou2_val = self.metric(yhat_val, y_val)
                # se agregan las métricas a una lista para luego ser promediadas
                loss_val_sublist.append(loss_val.data.item())
                acc_val_sublist.append(acc_val)
                iou1_val_sublist.append(iou1_val)
                iou2_val_sublist.append(iou2_val)
            
            # se promedian las métricas por epoch
            loss_mean_val = np.mean(loss_val_sublist)
            acc_mean_val = np.mean(acc_val_sublist)
            iou1_mean_val = np.mean(iou1_val_sublist)
            iou2_mean_val = np.mean(iou2_val_sublist)

            # se agregan los resultados para luego graficarlos
            loss_val_list = np.append(loss_val_list, loss_mean_val)
            acc_val_list = np.append(acc_val_list, acc_mean_val)
            iou1_val_list = np.append(iou1_val_list, iou1_mean_val)
            iou2_val_list = np.append(iou2_val_list, iou2_mean_val)

            # se muestra en pantalla las métricas por epoch durante el entrenamiento
            print(f"Epoch [{epoch+1}/{epochs}] - - - - - - - - -")
            print(f"Train Loss: {loss_mean:.2f} | Train Accuracy: {acc_mean:.2f} | Train IoU class 1: {iou1_mean:.2f} | Train IoU class 2: {iou2_mean:.2f}")
            print(f"Validation Loss: {loss_mean_val:.2f} | Validation Accuracy: {acc_mean_val:.2f} | Validation IoU class 1: {iou1_mean_val:.2f} | Validation IoU class 2: {iou2_mean_val:.2f}")
            print("- - - - - - - - - - - - - - - - - - - - - - -")

        # Las métricas se guarda en un diccionario
        self.metrics={
            "train_loss": loss_train_list,
            "train_accuracy": acc_train_list,
            "train_iou1" : iou1_train_list,
            "train_iou2" : iou2_train_list,
            "val_loss": loss_val_list,
            "val_accuracy": acc_val_list,
            "val_iou1" : iou1_val_list,
            "val_iou2" : iou2_val_list
        }