import numpy as np
import gc

def _remove_saturation(data_dict,constituents,saturation_mode):
    ET_thres = 4095  # saturated ET entries will always be dropped
    nof_MET = np.sum(constituents['MET'])
    nof_egamma = np.sum(constituents['EGAMMA'])
    nof_mu = np.sum(constituents['MUON'])
    nof_jet = np.sum(constituents['JET'])

    pt_thres = np.array([2047.5] * nof_MET + [255.5] * (nof_egamma + nof_mu) + [1023.5] * nof_jet) * 2
    
    # ET saturation treatment ...
    for event in data_dict.keys():
        mask = data_dict[event]["META"]["ET"] < ET_thres
        data_dict[event]["DATA"] = data_dict[event]["DATA"][mask]
        for k in data_dict[event]["META"].keys():
            data_dict[event]["META"][k] = data_dict[event]["META"][k][mask]
            
    # pT saturation treatment ....
    if saturation_mode == "drop":
        for event in data_dict.keys(): # In case of BKG the events are Train and Test
            mask = np.all(data_dict[event]["DATA"][:,:,0] < pt_thres, axis=1)
            data_dict[event]["DATA"] = data_dict[event]["DATA"][mask]

            for k in data_dict[event]["META"].keys():
                data_dict[event]["META"][k] = data_dict[event]["META"][k][mask]
    else:
        for event in data_dict.keys(): # In case of BKG the events are Train and Test
            mask = data_dict[event]["DATA"][:,:,0] < pt_thres
            data_dict[event]["DATA"] *= mask[:, :, None]
    del mask
    gc.collect()
    return data_dict