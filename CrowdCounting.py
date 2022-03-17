from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from TransCrowd.utils import dataframe_load_test

testList = np.loadtxt('VisDrone2020-CC/testlist.txt', delimiter=" ", dtype="str") 

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):

        if(annotations_file.split('/')[2].split('.')[0] in testList):
            df = dataframe_load_test(annotations_file)
        else:
            df = pd.read_csv(annotations_file, sep=",", header=None, index_col=False)

        df = df.reset_index(drop=True)
        
        df.columns = ['PICTURE', 'X', 'Y']

        df.sort_values(by=['PICTURE'])

        self.img_labels = df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, idx):  
        imageShortName = self.img_labels.iloc[idx, 0]
        
        # Se il nome è minore di 10, allora concateniamo 4 zeri, altrimenti 3
        if imageShortName < 10:
            imgFullName = "0000" + str(imageShortName) + ".jpg"
        else:
            imgFullName = "000" + str(imageShortName) + ".jpg"
            
        # Prendo l'immagine completa (Es. "00015.jpg")
        img_path = os.path.join(self.img_dir, imgFullName)
        image = read_image(img_path)
        
        # Lunghezza del file "annotations"
        n = len(self.img_labels)
        x_labels = list()
        y_labels = list()
        
        # Per tutto il file degli annotations, se il primo indice (che è relativo alla foto)
        # è uguale a quello che noi stiamo cercando, allora salvami tutte le variabili
        # in un Array da plottare successivamente. Se l'indice non è uguale, semplicemente
        # continua il flusso del ciclo for.
        for i in range (n):
            if (self.img_labels.iloc[i,0] == imageShortName):
                x_labels.append(self.img_labels.iloc[i,1])
                y_labels.append(self.img_labels.iloc[i,2])
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            x_labels = self.target_transform(x_labels)
            y_labels = self.target_transform(y_labels)
            
        return image, x_labels, y_labels

    def __len__(self):
        return len(self.img_labels)
        
    
def getRandomAnnotation():
    annotationValue = "0"
    randomAnnotation = 0
    
    randomAnnotation = random.randint(0, 112)

    if (randomAnnotation < 10):
        val = 4
    elif (randomAnnotation < 100):
        val = 3
    else:
        val = 2

    while (len(annotationValue) < val):
        annotationValue = "0" + annotationValue

    annotationValue += str(randomAnnotation)
    return annotationValue
    
# /--------------------------------- MAIN ---------------------------------\ #

annotationValue = getRandomAnnotation()

VisDroneDataSet = CustomDataset("VisDrone2020-CC/annotations/" + annotationValue + ".txt","VisDrone2020-CC/sequences/" + annotationValue + "/", None, None)

# Valore random da 0 fino all'ultima riga del file annotations
value = random.randint(0, len(VisDroneDataSet.img_labels))

# Ci prendiamo l'immagine, le coordinate x e y delle persone relative a quell'immagine
image, x_coordinates, y_coordinates = VisDroneDataSet.__getitem__(value)

plt.figure(0,figsize=(25,25))
plt.imshow(image.permute(1,2,0))

# Plottiamo i punti sullo schermo in base agli Array delle coordinate
plt.scatter(x_coordinates, y_coordinates, s=250, marker='.', c='r')

plt.title('Immagine: ')
plt.xlabel('Coordinate X')
plt.ylabel('Coordinate Y')

totalPeople = mpatches.Patch(color=None, label='Persone presenti: ' + str(len(x_coordinates)))
plt.legend(handles=[totalPeople], fontsize = 30)

plt.show()
