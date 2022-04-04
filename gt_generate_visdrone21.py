import os
import gdown
import shutil
import pandas as pd
from tqdm import tqdm
from lxml import etree
import cv2
import numpy as np
import h5py

# Set this percentage value for splitting dataset however you like.
# Training -> 80%
# Test     -> 20%
percentualeSplittingTrain = 80

datasetsDirectory = 'VisDrone2021-CC/'
VisDrone21_URL = 'https://drive.google.com/uc?id=' + '1EeY8kPAJn54PJuqCQvccQSoR7ke8HOf7' + '&export=download&confirm=t'
gt_files = []

def percentage(perc, totale):
  return round((perc * totale) / 100.0)

def downloadDataset(VisDrone21_URL):
    # Se non esiste la cartella dei datasets, la creiamo
    if not os.path.exists(datasetsDirectory):
        os.makedirs(datasetsDirectory)

    print("\nDownloading VisDrone2021-CC dataset...")

    gdown.download(VisDrone21_URL, 'VisDrone2021-CC.zip', quiet=False)

    print("\nExtracting datasets...")
    shutil.unpack_archive("VisDrone2021-CC.zip", datasetsDirectory)

    # Cleaning files
    print("\nCleaning files ...")

    os.remove("VisDrone2021-CC.zip")

    for file in os.listdir(os.getcwd()):
        if(file.endswith("tmp")):
            os.remove(os.path.join(os.getcwd(), file))

def sortFiles(Train):

    if (Train):
        directory = "Train"
    else:
        directory = "Val"

    gt_files = None

    i = 0
    gt_files = os.listdir(os.path.join(datasetsDirectory, "DroneRGBT", directory, "GT_"))

    for file in gt_files:
        f = file.split("R")[0]
        gt_files[i] = int(f)
        i += 1
    gt_files.sort()

    i = 0
    for file in gt_files:
        gt_files[i] = str(file) + "R.xml"
        i += 1
    return gt_files

def adjustXML():
    gt_files = None
    i = 0

    gt_files = os.listdir(os.path.join(datasetsDirectory, "DroneRGBT", "Train", "GT_"))

    for file in gt_files:
        f = file.split("R")[0]
        gt_files[i] = int(f)
        i += 1
    gt_files.sort()

    i = 0
    for file in gt_files:
        gt_files[i] = str(file) + "R.xml"
        i += 1

    i = 0

    percentageTrain = percentage(percentualeSplittingTrain, len(gt_files))

    try:
        shutil.rmtree("VisDrone2021-CC/DroneRGBT/test-challenge")
    except:
        # Probably already deleted
        pass

    # Val set
    if not os.path.exists(os.path.join(datasetsDirectory, "DroneRGBT", "Val")):
        os.makedirs(os.path.join(datasetsDirectory, "DroneRGBT", "Val"))
    if not os.path.exists(os.path.join(datasetsDirectory, "DroneRGBT", "Val", "GT_")):
        os.makedirs(os.path.join(datasetsDirectory, "DroneRGBT", "Val", "GT_"))
    if not os.path.exists(os.path.join(datasetsDirectory, "DroneRGBT", "Val", "TIR")):
        os.makedirs(os.path.join(datasetsDirectory, "DroneRGBT", "Val", "TIR"))

    print("-------------------------------------------")
    print("Adjusting XML files for VisDrone2021-CC ...")
    for gt_file in tqdm(gt_files):

        if(i < percentageTrain): # Train
            doc = etree.parse(os.path.join(datasetsDirectory, "DroneRGBT", "Train", "GT_", gt_file))
        else:                    # Test
            shutil.move(os.path.join(datasetsDirectory, "DroneRGBT", "Train", "GT_", gt_file), os.path.join(datasetsDirectory, "DroneRGBT", "Val", "GT_", gt_file))
            doc = etree.parse(os.path.join(datasetsDirectory, "DroneRGBT", "Val", "GT_", gt_file))

        root = doc.getroot()

        try:
            folderName = doc.find('.//folder').text
            doc.find('.//folder').text = "TIR"
        except:
            for elem in doc.findall("floder"):
                elem.tag = "folder"
            folderName = doc.find('.//folder').text
            doc.find('.//folder').text = "TIR"

        try:
            pathName = doc.find('.//path').text
            p = pathName.split("\\")
            p = p[len(p)-1]

            if(p.endswith(".xml")):
                p = p.replace("xml","jpg")
            elif(p.endswith(".json")):
                p = p.replace("json","jpg")

            if(i < percentageTrain): # Train
                doc.find('.//path').text = os.path.join(datasetsDirectory, "DroneRGBT", "Train", "TIR", p).replace("\\","/")
            else:                    # Test
                doc.find('.//path').text = os.path.join(datasetsDirectory, "DroneRGBT", "Val", "TIR", p).replace("\\","/")
                shutil.move(os.path.join(datasetsDirectory, "DroneRGBT", "Train", "TIR", p).replace("\\","/"), os.path.join(datasetsDirectory, "DroneRGBT", "Val", "TIR", p).replace("\\","/"))

        except:
            etree.SubElement(root, 'path')
            if(i < percentageTrain): # Train
                doc.find('.//path').text = os.path.join(datasetsDirectory, "DroneRGBT", "Train", "TIR", gt_file.replace("xml","jpg")).replace("\\","/")
            else:                    # Test
                doc.find('.//path').text = os.path.join(datasetsDirectory, "DroneRGBT", "Val", "TIR", gt_file.replace("xml","jpg")).replace("\\","/")
                shutil.move(os.path.join(datasetsDirectory, "DroneRGBT", "Train", "TIR", gt_file.replace("xml","jpg")).replace("\\","/"), os.path.join(datasetsDirectory, "DroneRGBT", "Val", "TIR", gt_file.replace("xml","jpg")).replace("\\","/"))
            pass

        if(i < percentageTrain): # Train
            doc.write(os.path.join(datasetsDirectory, "DroneRGBT", "Train", "GT_", gt_file))
        else:
            doc.write(os.path.join(datasetsDirectory, "DroneRGBT", "Val", "GT_", gt_file))

        i += 1

    print("-------------------------------------------")

def splitDataset():
    # Se non esiste la cartella delle immagini splittate, la creiamo
    if not os.path.exists(os.path.join(datasetsDirectory,"DroneRGBT","Train","images_crop")):
        os.makedirs(os.path.join(datasetsDirectory,"DroneRGBT","Train","images_crop"))
    if not os.path.exists(os.path.join(datasetsDirectory,"DroneRGBT","Train","gt_density_map")):
        os.makedirs(os.path.join(datasetsDirectory,"DroneRGBT","Train","gt_density_map"))

    if not os.path.exists(os.path.join(datasetsDirectory,"DroneRGBT","Val","images")):
        os.makedirs(os.path.join(datasetsDirectory,"DroneRGBT","Val","images"))
    if not os.path.exists(os.path.join(datasetsDirectory,"DroneRGBT","Val","gt_density_map")):
        os.makedirs(os.path.join(datasetsDirectory,"DroneRGBT","Val","gt_density_map"))

    print("Splitting files for VisDrone2021-CC...")
    print("- Starting training set...")

    pathErrors = 0

    gt_filesTrain = sortFiles(Train=True)
    gt_filesTest = sortFiles(Train=False)

    batch_barTrain = tqdm(total=len(gt_filesTrain), desc="Splitting", position=0, miniters=5)
    batch_barTest = tqdm(total=len(gt_filesTest), desc="Splitting", position=0, miniters=5)

    for gt_file in gt_filesTrain:

        dfVisDrone = pd.DataFrame(columns=['PATH','NAME','WIDTH','HEIGHT','X','Y'])

        # Train
        doc = etree.parse(os.path.join(datasetsDirectory, "DroneRGBT", "Train", "GT_", gt_file))

        root = doc.getroot()

        for size in root.findall('size'):
            width = size.find('width').text
            height = size.find('height').text

        for size in root.findall('object'):
            name = size.find('name').text

            try:
                x = size.find('point/x').text
                y = size.find('point/y').text
            except:
                x = size.find('bndbox/xmin').text
                y = size.find('bndbox/ymin').text
                #xmax = size.find('bndbox/xmax').text
                #ymax = size.find('bndbox/ymax').text

            dfVisDrone = pd.concat([dfVisDrone,
                pd.DataFrame.from_records([{
                    'PATH': doc.find('.//path').text,
                    'NAME': name,
                    'WIDTH': width,
                    'HEIGHT': height,
                    'X': x,
                    'Y': y }
                ])
            ])
        
        path = (doc.find('.//path').text).replace("/","\\")

        if(os.path.isfile(path)):
            img_data = cv2.imread(path)
        else:
            path = path.replace("Train", "Val")
            img_data = cv2.imread(path)
            pathErrors += 1
            pass
        
        kpoint = np.zeros((img_data.shape[0], img_data.shape[1]))
        d_map = (np.zeros((img_data.shape[0], img_data.shape[1])) + 255).astype(np.uint8)
            
        for index, row in dfVisDrone.iterrows():
            gt_x = int(row['X'])
            gt_y = int(row['Y'])

            if gt_y < kpoint.shape[0] and gt_x < kpoint.shape[1]:
                kpoint[int(gt_y), int(gt_x)] = 1
                d_map[int(gt_y)][int(gt_x)] = d_map[int(gt_y)][int(gt_x)] - 255

        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        kpoint = kpoint.astype(np.uint8)

        gt_count = np.sum(kpoint).astype(np.float32)

        # Salvataggio immagini ridimensionate in 
        # 384x384 in VisDrone2021-CC/DroneRGBT/Train/images_crop/1R.jpg etc.
        # -- Solo per visualizzazione --
        imgName = dfVisDrone.iloc[0,0].split("/")
        imgName = imgName[len(imgName) - 1]

        # Train
        saveImg = os.path.join(datasetsDirectory,"DroneRGBT","Train","images_crop", imgName).replace("/","\\")

        cv2.imwrite(saveImg, img_data)

        # Salvataggio immagini ridimensionate in 384x384 all'interno di
        # VisDrone2020-CC/train_data/gt_density_map/00001_00001_0_0.h5 etc.
        # Qui viene salvato il gt_count
        # Train
        h5_path = saveImg.replace('images_crop', 'gt_density_map').replace('.jpg', '.h5')

        with h5py.File(h5_path, 'w') as hf:
            hf['gt_count'] = gt_count
            hf['img_data'] = img_data
            hf['kpoint'] = kpoint

        batch_barTrain.update(1)

    print("---------------------------")
    print("- Starting validation set...")

    gt_file = None
    for gt_file in gt_filesTest:

        dfVisDrone = pd.DataFrame(columns=['PATH','NAME','WIDTH','HEIGHT','X','Y'])

        # Test
        doc = etree.parse(os.path.join(datasetsDirectory, "DroneRGBT", "Val", "GT_", gt_file))

        root = doc.getroot()

        for size in root.findall('size'):
            width = size.find('width').text
            height = size.find('height').text

        for size in root.findall('object'):
            name = size.find('name').text

            try:
                x = size.find('point/x').text
                y = size.find('point/y').text
            except:
                x = size.find('bndbox/xmin').text
                y = size.find('bndbox/ymin').text
                #xmax = size.find('bndbox/xmax').text
                #ymax = size.find('bndbox/ymax').text

            dfVisDrone = pd.concat([dfVisDrone,
                pd.DataFrame.from_records([{
                    'PATH': doc.find('.//path').text,
                    'NAME': name,
                    'WIDTH': width,
                    'HEIGHT': height,
                    'X': x,
                    'Y': y }
                ])
            ])
        
        path = (doc.find('.//path').text).replace("/","\\")
        if(os.path.isfile(path)):
            img_data = cv2.imread(path)
        else:
            path = path.replace("Val", "Train")
            img_data = cv2.imread(path)
            pathErrors += 1
            pass
        
        kpoint = np.zeros((img_data.shape[0], img_data.shape[1]))
        d_map = (np.zeros((img_data.shape[0], img_data.shape[1])) + 255).astype(np.uint8)
            
        for index, row in dfVisDrone.iterrows():
            gt_x = int(row['X'])
            gt_y = int(row['Y'])

            if gt_y < kpoint.shape[0] and gt_x < kpoint.shape[1]:
                kpoint[int(gt_y), int(gt_x)] = 1
                d_map[int(gt_y)][int(gt_x)] = d_map[int(gt_y)][int(gt_x)] - 255

        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        kpoint = kpoint.astype(np.uint8)

        gt_count = np.sum(kpoint).astype(np.float32)

        # Salvataggio immagini ridimensionate in 
        # 384x384 in VisDrone2021-CC/DroneRGBT/Train/images_crop/1R.jpg etc.
        # -- Solo per visualizzazione --
        imgName = dfVisDrone.iloc[0,0].split("/")
        imgName = imgName[len(imgName) - 1]

        # Test
        saveImg = os.path.join(datasetsDirectory,"DroneRGBT","Val","images", imgName).replace("/","\\")

        cv2.imwrite(saveImg, img_data)

        # Salvataggio immagini ridimensionate in 384x384 all'interno di
        # VisDrone2020-CC/train_data/gt_density_map/00001_00001_0_0.h5 etc.
        # Qui viene salvato il gt_count

        # Test
        h5_path = saveImg.replace('images', 'gt_density_map').replace('.jpg', '.h5')

        with h5py.File(h5_path, 'w') as hf:
            hf['gt_count'] = gt_count
            hf['img_data'] = img_data
            hf['kpoint'] = kpoint

        batch_barTest.update(1)
        
    print("-------------------------------------------")
    print(F"Path errors: {pathErrors}")

# ========================= D O W N L O A D ==========================

downloadDataset(VisDrone21_URL)

# ======================== A D J U S T   X M L =======================

adjustXML()

# ======================== S P L I T T I N G  ========================

splitDataset()

print("---------------------------")
print("Everything done. Exiting..")
