import numpy as np
import h5py
import gc


# Only get_data_background and get_data_signal should be used by the end user _* functions are helpers and not meant to be
# individualy

def _get_raw_background(file_path,train_sector,test_sector,object_ranges,verbose = False):
    
    """
    file_path: Path to the original H5 file
    train_sector: Portion of the file that will be used for training
    test_sector: Portion of the file that will be used for evaluation
    """
    f = h5py.File(file_path,"r")
    if verbose:
        print("H5 file opened successfully")
    
    dataset = f["full_data_cyl"]
    dat_met = dataset[train_sector[0]:train_sector[1],object_ranges["met"],:]
    dat_egs = dataset[train_sector[0]:train_sector[1],object_ranges["egs"][0]:object_ranges["egs"][1],:]
    dat_muons = dataset[train_sector[0]:train_sector[1],object_ranges["muons"][0]:object_ranges["muons"][1],:]
    dat_jets = dataset[train_sector[0]:train_sector[1],object_ranges["jets"][0]:object_ranges["jets"][1],:]
    
    if verbose:
        print("Train_Data read successful")
    # Few meta datas ....
    ET = f["ET"][train_sector[0]:train_sector[1]]
    HT = f["HT"][train_sector[0]:train_sector[1]]
    PU = f["event_info"][train_sector[0]:train_sector[1],:]
    L1_bits = f["L1bit"][train_sector[0]:train_sector[1]]
    
    if verbose:
        print("Train_Meta read successful")
        
    return_dict = {}
    return_dict["Train"]={
    "DATA":{
        "MET":dat_met,
        "EGAMMA":dat_egs,
        "MUON": dat_muons,
        "JET": dat_jets},
     "META":{
        "ET":ET,
        "HT":HT,
        "PU":PU,
        "L1bits":L1_bits}}
    del dat_met, dat_egs, dat_muons, dat_jets, ET, HT, PU, L1_bits
    gc.collect()
    
    if verbose:
        print("Registry and Garbage cleaning successful")
    
    dat_met = dataset[test_sector[0]:test_sector[1],object_ranges["met"],:]
    dat_egs = dataset[test_sector[0]:test_sector[1],object_ranges["egs"][0]:object_ranges["egs"][1],:]
    dat_muons = dataset[test_sector[0]:test_sector[1],object_ranges["muons"][0]:object_ranges["muons"][1],:]
    dat_jets = dataset[test_sector[0]:test_sector[1],object_ranges["jets"][0]:object_ranges["jets"][1],:]
    
    if verbose:
        print("Test_Data read successful")
    
    # Few meta datas ....
    ET = f["ET"][test_sector[0]:test_sector[1]]
    HT = f["HT"][test_sector[0]:test_sector[1]]
    PU = f["event_info"][test_sector[0]:test_sector[1],:]
    L1_bits = f["L1bit"][test_sector[0]:test_sector[1]]

    if verbose:
        print("Test_Meta read successful")
        
    return_dict["Test"]={
    "DATA":{
        "MET":dat_met,
        "EGAMMA":dat_egs,
        "MUON": dat_muons,
        "JET": dat_jets},
     "META":{
        "ET":ET,
        "HT":HT,
        "PU":PU,
        "L1bits":L1_bits}}
    del dat_met, dat_egs, dat_muons, dat_jets, ET, HT, PU, L1_bits
    gc.collect()
    f.close()
    if verbose:
        print("Registry and Garbage cleaning successful")
    return return_dict
    
def _stripper_bkg(data_dict,constituents,verbose = False):
    if verbose:
        print("Starting ...")
    for case in ["Train","Test"]:
        for k in data_dict[case]["DATA"].keys():
            if k !="MET":
                data_dict[case]["DATA"][k] = data_dict[case]["DATA"][k][:,constituents[k],:]
            else:
                 data_dict[case]["DATA"][k] = data_dict[case]["DATA"][k][:,None,:]
    if verbose:
        print("Successful ...")
    gc.collect()
    if verbose:
        print("Garbage cleaning successful")
    return data_dict

def get_data_background(config,verbose = False):
    
    file_path = config["file_path"]
    train_sector = config["train_sector"]
    test_sector = config["test_sector"]
    object_ranges = config["object_ranges"]
    constituents = config["constituents"]
    
    data = _get_raw_background(file_path = file_path,
                               train_sector = train_sector,
                               test_sector = test_sector,
                               object_ranges = object_ranges,
                               verbose=verbose)
    data = _stripper_bkg(data_dict=data,
                     constituents=constituents,
                     verbose=verbose)
    
    new_dict = {}
    
    for case in data.keys():
        new_dict[case] = {
            "META":data[case]["META"],
            "DATA":np.concatenate(list(data[case]["DATA"].values()),axis = 1)
        }
    del data
    gc.collect()
    return new_dict

def _get_raw_signal(file_path,MAX_NUM_SIGNALS,object_ranges,verbose = False):
    return_dict = {}

    data = h5py.File(file_path,"r")
    signal_names = [k for k in data.keys() if data[k].ndim == 3]
    for signal_name in signal_names:
        dataset = data[signal_name]

        dat_met = dataset[:MAX_NUM_SIGNALS,object_ranges["met"],:]
        dat_egs = dataset[:MAX_NUM_SIGNALS,object_ranges["egs"][0]:object_ranges["egs"][1],:]
        dat_muons = dataset[:MAX_NUM_SIGNALS,object_ranges["muons"][0]:object_ranges["muons"][1],:]
        dat_jets = dataset[:MAX_NUM_SIGNALS,object_ranges["jets"][0]:object_ranges["jets"][1],:]

        ET = data[f'{signal_name}_ET'][:MAX_NUM_SIGNALS]  # type:ignore
        HT = data[f'{signal_name}_HT'][:MAX_NUM_SIGNALS]  # type:ignore
        PU = data[f'{signal_name}_nPV'][:MAX_NUM_SIGNALS]  # type:ignore
        L1_bits = data[f'{signal_name}_l1bit'][:MAX_NUM_SIGNALS]  # type:ignore

        return_dict[signal_name] = {
            "DATA":{
                "MET":dat_met,
                "EGAMMA":dat_egs,
                "MUON": dat_muons,
                "JET": dat_jets},
             "META":{
                "ET":ET,
                "HT":HT,
                "PU":PU,
                "L1bits":L1_bits}}
        del dat_met, dat_egs, dat_muons, dat_jets, ET, HT, PU, L1_bits
        gc.collect()
    data.close()
    return return_dict
    
def _stripper_sig(data_dict,constituents,verbose = False):
    if verbose:
        print("Starting ...")
    for case in data_dict.keys():
        for k in data_dict[case]["DATA"].keys():
            if k !="MET":
                data_dict[case]["DATA"][k] = data_dict[case]["DATA"][k][:,constituents[k],:]
            else:
                 data_dict[case]["DATA"][k] = data_dict[case]["DATA"][k][:,None,:]
    if verbose:
        print("Successful ...")
    gc.collect()
    if verbose:
        print("Garbage cleaning successful")
    return data_dict

def get_data_signal(config,verbose = False):
    
    file_path = config["file_path"]
    MAX_NUM_SIGNALS = config["MAX_NUM_SIGNALS"]
    object_ranges = config["object_ranges"]
    constituents = config["constituents"]
    
    data = _get_raw_signal(file_path=file_path,
                            MAX_NUM_SIGNALS=MAX_NUM_SIGNALS,
                            object_ranges=object_ranges,
                            verbose=verbose
                          )
    
    data = _stripper_sig(data_dict=data,
                     constituents=constituents,
                     verbose=verbose)
    
    new_dict = {}
    
    for case in data.keys():
        new_dict[case] = {
            "META":data[case]["META"],
            "DATA":np.concatenate(list(data[case]["DATA"].values()),axis = 1)
        }
    del data
    gc.collect()
        
    return new_dict