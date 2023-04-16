from concurrent.futures import ProcessPoolExecutor
import shutil

def load_data(path):
    f = open(path, 'r')
    lines = f.readlines()
    # print(len(lines))
    for line in lines:

        source_path = r'C:/Users/zhouz/Desktop/'
        target_path = 'RAF_Data/'
        if str(line.split('_')[2].split('/')[0]) == 'train':

            target_path = target_path + 'new_train'
            if int(line.split(' ')[1]) == 1:
                target_path = target_path + '\\0'
            elif int(line.split(' ')[1]) == 2:
                target_path = target_path + '\\1'
            elif int(line.split(' ')[1]) == 3:
                target_path = target_path + '\\2'
            elif int(line.split(' ')[1]) == 4:
                target_path = target_path + '\\0'
            elif int(line.split(' ')[1]) == 5:
                target_path = target_path + '\\4'
            elif int(line.split(' ')[1]) == 6:
                target_path = target_path + '\\5'
            else:
                target_path = target_path + '\\6'
            print(target_path)
            shutil.copy(source_path + str(line.split(' ')[0]), target_path)
        else:
            target_path = target_path + 'new_val'
            if int(line.split(' ')[1]) == 1:
                target_path = target_path + '\\0'
            elif int(line.split(' ')[1]) == 2:
                target_path = target_path + '\\1'
            elif int(line.split(' ')[1]) == 3:
                target_path = target_path + '\\2'
            elif int(line.split(' ')[1]) == 4:
                target_path = target_path + '\\0'
            elif int(line.split(' ')[1]) == 5:
                target_path = target_path + '\\4'
            elif int(line.split(' ')[1]) == 6:
                target_path = target_path + '\\5'
            else:
                target_path = target_path + '\\6'
            print(target_path)
            shutil.copy(source_path + str(line.split(' ')[0]), target_path)
    print('it is over')

if __name__=="__main__":
    train = r'C:\Users\zhouz\Desktop\trainvalid\new_train_label_dic.txt'
    test = r'C:\Users\zhouz\Desktop\trainvalid\new_test_label_dic.txt'
    load_data(test)