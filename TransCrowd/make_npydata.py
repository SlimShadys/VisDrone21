import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Configuration train phase")
parser.add_argument("-d", "--dataset", type=str, default="VisDrone", choices=["VisDrone", "ShanghaiTech"], help='Choose the desired dataset')
args = parser.parse_args()

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

dataset = args.dataset

if(dataset == 'VisDrone'):
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
elif(dataset == 'ShanghaiTech'):
    try:
        shanghaiAtrain_path = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/part_A_final/train_data/images_crop/'
        shanghaiAtest_path = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/part_A_final/test_data/images_crop/'

        train_list = []
        for filename in os.listdir(shanghaiAtrain_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(shanghaiAtrain_path + filename)

        train_list.sort()
        np.save('./npydata/ShanghaiA_train.npy', train_list)

        test_list = []
        for filename in os.listdir(shanghaiAtest_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(shanghaiAtest_path + filename)
        test_list.sort()
        np.save('./npydata/ShanghaiA_test.npy', test_list)

        print("Generate ShanghaiA image list successfully")
        print("\t- Train list: " + str(len(train_list)))
        print("\t- Test list: " + str(len(test_list)))
    except:
        print("The ShanghaiA dataset path is wrong. Please check you path.")

    try:
        shanghaiBtrain_path = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/part_B_final/train_data/images_crop/'
        shanghaiBtest_path = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/part_B_final/test_data/images_crop/'

        train_list = []
        for filename in os.listdir(shanghaiBtrain_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(shanghaiBtrain_path + filename)
        train_list.sort()
        np.save('./npydata/ShanghaiB_train.npy', train_list)

        test_list = []
        for filename in os.listdir(shanghaiBtest_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(shanghaiBtest_path + filename)
        test_list.sort()
        np.save('./npydata/ShanghaiB_test.npy', test_list)
        print("Generate ShanghaiB image list successfully")
        print("\t- Train list: " + str(len(train_list)))
        print("\t- Test list: " + str(len(test_list)))
    except:
        print("The ShanghaiB dataset path is wrong. Please check your path.")
else:
    print("Please choose a dataset in order to make a numpy image list.")
    print("- Ex. 'python make_npydata --dataset 'VisDrone''")
    exit(0)
