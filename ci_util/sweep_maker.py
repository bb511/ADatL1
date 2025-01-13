### This script prepares the inital setup of the codes including setting up the environment, EOS space and AFS directories.
import sys
import os
import yaml
import numpy
import random, string

env_command = "conda activate /afs/cern.ch/user/m/mlflow/public/winter"

afs_path = sys.argv[1]
axo_eos_path = sys.argv[2]
ntuple_eos_path = sys.argv[3]

lib_path = os.path.join(afs_path, "l1_anomaly_ae")

os.makedirs(os.path.join(afs_path,"CONFIG_DIR"))

config_dir = os.path.join(afs_path,"CONFIG_DIR")

def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

def get_code():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))

master = yaml.load(open(f"{lib_path}/axo/utilities/config.yml","r"), Loader=yaml.Loader)

beta_values = [i/100 for i in range(10,90,5)]
background_path = os.path.join(ntuple_eos_path,"ZB.h5")
signal_path = os.path.join(ntuple_eos_path,"BSM.h5")

for beta in beta_values:
    slave = master.copy()
    run_name = get_code()
    
    slave["data_config"]["Read_configs"]["BACKGROUND"]["file_path"] = background_path
    slave["data_config"]["Read_configs"]["SIGNAL"]["file_path"] = signal_path
    slave["data_config"]["Processed_data_path"] = os.path.join(axo_eos_path,"Data_file",f"{run_name}.h5")
    slave["train"]["VAE_config"]["beta"] = beta
    slave["store"]["lite_path"] = os.path.join(axo_eos_path,"Lite_file",f"{run_name}_lte.h5")
    slave["store"]["complete_path"] = os.path.join(axo_eos_path,"Complete_file",f"{run_name}_complete.h5")
    slave["store"]["build_model_path"] = os.path.join(axo_eos_path,"Build_model",f"{run_name}_build.h5")
    slave["store"]["temp_path"] = os.path.join(axo_eos_path)
    slave["report"]["html_report"]["path"] = os.path.join(axo_eos_path,"html_report",f"{run_name}.html")
    slave["report"]["pdf_report"]["path"] = os.path.join(axo_eos_path,"pdf_report",f"{run_name}.pdf")
    
    slave["train"]["common"]["n_epochs"] = 5 ## To be commented out later !!!!

    #####
    with open(f'{config_dir}/{run_name}.yml', 'w') as outfile:
        yaml.dump(slave, outfile, default_flow_style=False, sort_keys=False)
