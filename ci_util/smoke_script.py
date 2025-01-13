import sys
import os
import glob

afs_path = sys.argv[1]
config_dir = sys.argv[2]

lib_path = os.path.join(afs_path, "l1_anomaly_ae")
sys.path.append(lib_path)
import axo
import yaml

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

if __name__ == "__main__":
    
    config_file_list = glob.glob(os.path.join(config_dir,"*.yml"))
    print("Number of Config Files found",{len(config_file_list)})

    master = yaml.load(open(os.path.join(lib_path,"axo/utilities/config.yml"),"r"), Loader=yaml.Loader)
    exit_stat = 0 

    for file in config_file_list:
        slave = yaml.load(open(os.path.join(lib_path,file),"r"), Loader=yaml.Loader)

        if axo.utilities.check_compartibility(master=master,slave=slave) == 1:
            print(f"Configuration file {file} passed")
            continue
        else:
            print(f"Configuration file {file} failed, terminating....")
            exit_stat=1
    
    exit(exit_stat)
