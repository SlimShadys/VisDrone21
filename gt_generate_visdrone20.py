import numpy as np
import cv2
import os
import h5py
from scipy.ndimage.filters import gaussian_filter

path = 'VisDrone2020-CC/annotations'
trainDataDirectory = 'VisDrone2020-CC/train_data/images/'
testDataDirectory = 'VisDrone2020-CC/test_data/images/'

if not os.path.exists(trainDataDirectory):
    os.makedirs(trainDataDirectory)
    os.makedirs(trainDataDirectory.replace('images','gt_density_show'))
    os.makedirs(trainDataDirectory.replace('images','gt_density_map'))
    os.makedirs(trainDataDirectory.replace('images','gt_fidt_map'))
    os.makedirs(trainDataDirectory.replace('images','images_crop'))
    
if not os.path.exists(testDataDirectory):
    os.makedirs(testDataDirectory)
    os.makedirs(testDataDirectory.replace('images','gt_density_map'))
    
gt_paths=[]
testList = np.loadtxt('VisDrone2020-CC/testlist.txt', delimiter=" ", dtype="str") 
trainList = np.loadtxt('VisDrone2020-CC/trainlist.txt', delimiter=" ", dtype="str")

for k in range(1,112):
        if k < 10:
            gt_paths.append(path + '/0000' + str(k) + '.txt')
        elif k < 100:
            gt_paths.append(path + '/000' + str(k) + '.txt')
        else:
            gt_paths.append(path + '/00' + str(k) + '.txt')

#for path in path_sets:
#    for img_path in glob.glob(os.path.join(path, '*.txt')):
#        gt_paths.append(img_path)
gt_paths.sort()
print("There are " + str(len(gt_paths)) + " directories to retrieve images from")

#=============================================================================

for i in range(len(gt_paths)):

    # Mi prendo i path contenenti gli annotations:
    # - VisDrone2020-CC/annotations\00001.txt
    # - VisDrone2020-CC/annotations\00002.txt
    # - VisDrone2020-CC/annotations\00003.txt    
    gt_path = gt_paths[i]

    print(str(i) + ") " + gt_path.replace('annotations','sequences'))

    img_path_root = gt_path.replace('annotations', 'sequences').split('.')[0]
    
    if(gt_path.split('/')[2].split('.')[0] in trainList):
        with open(gt_path, "r") as f:   # Apertura file
            gt_file = f.readlines()     # Lettura file
        gt_file = np.array(gt_file)     # Scrivo in un array, tutte le coordinate delle persone

    for k in range(1, 31):
        if k < 10:
            img_path = img_path_root + '/' + '0000' + str(k) + '.jpg'
        else:
            img_path = img_path_root + '/' + '000' + str(k) + '.jpg'

        img_path = img_path.replace("\\","/")
        firstSplit = img_path.split('/')[2]
        secondSplit = img_path.split('/')[3]
        save_filename = firstSplit + '_' + secondSplit

        Img_data = cv2.imread(img_path)
        
        rate_h = 768.0 / Img_data.shape[0] # Altezza
        rate_w = 1156.0 / Img_data.shape[1] # Larghezza
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_w, fy=rate_h)
        
        kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
        d_map = (np.zeros((Img_data.shape[0], Img_data.shape[1])) + 255).astype(np.uint8)

        # Da capire se questo influisce sulle immagini di Test o meno
        for l in range(len(gt_file)):
           fname = int(gt_file[l].split(',')[0])
           
           if fname == k:    
               gt_x = int(gt_file[l].split(',')[1]) * rate_w
               gt_y = int(gt_file[l].split(',')[2]) * rate_h

               if gt_y < kpoint.shape[0] and gt_x < kpoint.shape[1]:
                    kpoint[int(gt_y), int(gt_x)] = 1
                    d_map[int(gt_y)][int(gt_x)] = d_map[int(gt_y)][int(gt_x)] - 255
        
        # Salvataggio immagini ridimensionate in 1156x768 e B&W
        # in VisDrone2020-CC/train_data/images/00001_00001.jpg etc.
        # solo per visualizzazione
        save_img = trainDataDirectory + save_filename
        #Img_data = cv2.cvtColor(Img_data, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite(save_img, Img_data)
        
        # Salvataggio immagini ridimensionate in 1156x768
        # in VisDrone2020-CC/train_data/gt_fidt_map/00001_00001.h5 etc.
        # dove mi salvo i kpoint e la density_map
        density_map = gaussian_filter(kpoint, 4)
        kpoint = kpoint.astype(np.uint8)
        density_map = density_map.astype(np.float64)
        with h5py.File(save_img.replace('.jpg', '.h5').replace('images', 'gt_fidt_map'), 'w') as hf:
            hf['kpoint'] = kpoint
            hf['density_map'] = density_map

        # Salvataggio immagini ridimensionate in 
        # 1156x768 a cui viene applicato il GaussianFilter
        # in VisDrone2020-CC/train_data/gt_density_show/00001_00001.jpg etc.
        # Solo per visualizzazione.
        density_map = density_map
        density_map = density_map / np.max(density_map) * 255
        density_map = density_map.astype(np.uint8)
        density_map = cv2.applyColorMap(density_map, 2)
        cv2.imwrite(save_img.replace('images', 'gt_density_show').replace('jpg', 'jpg'), density_map)
        
        #-----------------------#
        
        # Se ci troviamo in un file della trainList, allora ci prendiamo 6 sub-images
        # e per ognuna di queste mi calcolo il ground_truth
        if(gt_path.split('/')[2].split('.')[0] in trainList):
            height, width = Img_data.shape[0], Img_data.shape[1]
            m = int(width / 384)
            n = int(height / 384)
                        
            # Se il file che stiamo trattando si trova all'interno della trainList
            # salviamolo nella cartella opportuna "VisDrone2020-CC/train_data/"
            for i in range(0, m):
                for j in range(0, n):
                    # Per ogni immagine, la divido in parti uguali da 384x384 ciascuno
                    # sia per l'altezza, sia per la larghezza.
                    crop_img = Img_data[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
                    crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384]
                    gt_count = np.sum(crop_kpoint) # ---> Il ground_truth sarà la somma
                                                   #      di tutti i gt di tutte le sub-immagini
                                                   
                    # Salvataggio immagini ridimensionate in 384x384 all'interno di
                    # VisDrone2020-CC/train_data/images_crop/00001/00001_0_0.jpg etc.
                    # Solo per visualizzazione
                    save_img = trainDataDirectory.replace('images', 'images_crop') + save_filename.split("_")[0] + '_' + save_filename.split("_")[1].split(".")[0] + '_' + str(i) + '_' + str(j) + '.jpg' 
                    cv2.imwrite(save_img, crop_img)

                    # Salvataggio immagini ridimensionate in 384x384 all'interno di
                    # VisDrone2020-CC/train_data/gt_density_map/00001/00001_0_0.h5 etc.
                    # Qui viene salvato il gt_count
                    h5_path = save_img.replace('.jpg', '.h5').replace('images_crop', 'gt_density_map')
                    with h5py.File(h5_path, 'w') as hf:
                        hf['gt_count'] = gt_count
                                 
                    print("\tPath: " + save_img)
                    print("\t\t- Ground truth count for " + h5_path + ": " + str(gt_count))
        
        # Se il file che stiamo trattando non si trova all'interno della trainList
        # salviamolo nella cartella opportuna "VisDrone2020-CC/test_data/"
        else:
            
            # Salvataggio immagini ridimensionate in 1156x768 all'interno di
            # in VisDrone2020-CC/test_data/images/00011/00001.jpg etc.
            # Solo per visualizzazione
            save_img = testDataDirectory + save_filename.split("_")[0] + '_' + save_filename.split("_")[1].split(".")[0] + '_' + '.jpg' 
            cv2.imwrite(save_img, Img_data)
            
            gt_count = np.sum(kpoint)
            h5_path = save_img.replace('images', 'gt_density_map').replace('.jpg', '.h5')
            with h5py.File(h5_path, 'w') as hf:
                hf['gt_count'] = gt_count
                
            print("\tPath: " + save_img)
            print("\t\t- Ground truth count for " + h5_path + ": " + str(gt_count)) 

    print("------------------------------------")
    # print(img_path_root)