import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

try:
    VisDroneTrain_path = '../VisDrone2020-CC/train_data/gt_density_map/'
    VisDroneTest_path = '../VisDrone2020-CC/test_data/gt_density_map/'

    train_list = []
    for filename in os.listdir(VisDroneTrain_path):
        if filename.split('.')[1] == 'h5':
            train_list.append(VisDroneTrain_path + filename)

    train_list.sort()
    np.save('./npydata/visDrone_train.npy', train_list)
    
    test_list = []
    for filename in os.listdir(VisDroneTest_path):
        if filename.split('.')[1] == 'h5':
            test_list.append(VisDroneTest_path + filename)
    test_list.sort()
    np.save('./npydata/visDrone_test.npy', test_list)

    print("Generate VisDrone image list successfully")
    print("\t- Train list: " + str(len(train_list)))
    print("\t- Test list: " + str(len(test_list)))
except:
    print("The VisDroneTrain dataset path is wrong. Please check you path.")
