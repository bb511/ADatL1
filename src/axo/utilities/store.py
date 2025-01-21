import h5py
import random
import time
import tensorflow as tf
import os
import json
import numpy as np

def save_inside_h5(model,h5,temp_path):
    time_part = str(time.time_ns())
    rand_part = str(random.randint(0,1e20))
    name = f"{time_part}_{rand_part}.h5"
    path = os.path.join(temp_path,name)
    tf.keras.models.save_model(model, path) 
    h_ = h5py.File(path,"r")
    h_.copy(source=h_["model_weights"],dest=h5)
    h_.close()
    os.remove(path)
    
    
def store_axo(config,model,model_trim,axo_man,dist_plot,history_dict):
    
    config_ser = json.dumps(config)
    skip_flag = config["store"]["skip_saving_complete"]
    
    if not skip_flag:
        complete = h5py.File(config["store"]["complete_path"],"w")
        complete.attrs["config"] = config_ser
    lite = h5py.File(config["store"]["lite_path"],"w")
    lite.attrs["config"] = config_ser
    
    ###############################################################################
    if not skip_flag:
        #### Storing the data into the complete --->
        c_data = complete.create_group("data")
        processed_data = h5py.File(config["data_config"]["Processed_data_path"],"r")
        for k in processed_data.keys():
            processed_data.copy(source=processed_data[k],dest=c_data)
    ###############################################################################
    # Storing the entire model in complete and only trimmed encoder in lite
    if not skip_flag:
        c_model = complete.create_group("model")

        c_model.create_group("encoder")
        c_model.create_group("decoder")
        c_model.create_group("trimmed_encoder")


    l_model = lite.create_group("model")

    l_model.create_group("trimmed_encoder")

    if not skip_flag:
        save_inside_h5(model=model.encoder,h5=c_model["encoder"],temp_path=config["store"]["temp_path"])
        save_inside_h5(model=model.decoder,h5=c_model["decoder"],temp_path=config["store"]["temp_path"])
        save_inside_h5(model=model_trim,h5=c_model["trimmed_encoder"],temp_path=config["store"]["temp_path"])

    save_inside_h5(model=model_trim,h5=l_model["trimmed_encoder"],temp_path=config["store"]["temp_path"])
    
    # Storing the build model using keras.save for FW
    tf.keras.models.save_model(model_trim, config["store"]["build_model_path"])

    # Model storing complete
    ###############################################################################
    
    if not skip_flag:
        c_results = complete.create_group("results")
    l_results = lite.create_group("results")
    
    if not skip_flag:
        c_thres = c_results.create_group("threshold")
    l_thres = l_results.create_group("threshold")

    # Threshold storage
    ###############################################################################
    if not skip_flag:
        c_thres.create_dataset(name="key",data = np.array(list(axo_man.threshold.keys()), dtype=h5py.special_dtype(vlen=str)))
        c_thres.create_dataset(name="value",data= np.array(list(axo_man.threshold.values())))

    l_thres.create_dataset(name="key",data = np.array(list(axo_man.threshold.keys()), dtype=h5py.special_dtype(vlen=str)))
    l_thres.create_dataset(name="value",data= np.array(list(axo_man.threshold.values())))
    ###############################################################################

    # AXO SCORE STORAGE
    if not skip_flag:
        c_axo = c_results.create_group("axo_scores")
    l_axo = l_results.create_group("axo_scores")

    for thres in axo_man.threshold.keys():
        if not skip_flag:
            c_sc_thres = c_axo.create_group(thres)
        l_sc_thres = l_axo.create_group(thres)
        df = axo_man.get_score(thres)
        df = df.to_dict()
        for k in df.keys():
            if "name" in k.lower(): ## This is done since HDF5 cannot handle string properly, moreover one of the column only contain string
                if not skip_flag:
                    c_sc_thres.create_dataset(name=k,data=np.array(list(df[k].values()), dtype=h5py.special_dtype(vlen=str)))
                l_sc_thres.create_dataset(name=k,data=np.array(list(df[k].values()), dtype=h5py.special_dtype(vlen=str)))
            else:
                if not skip_flag:
                    c_sc_thres.create_dataset(name=k,data=np.array(list(df[k].values())))
                l_sc_thres.create_dataset(name=k,data=np.array(list(df[k].values())))
    ###############################################################################

    # Score histogram storage
    if not skip_flag:
        c_hist = c_results.create_group("score_hist")
    l_hist = l_results.create_group("score_hist")

    ### Background histogram ....
    if not skip_flag:
        c_hist_bkg = c_hist.create_group("background")
    l_hist_bkg = l_hist.create_group("background")

    count_, bin_ = dist_plot.background_hist
    
    if not skip_flag:
        c_hist_bkg.create_dataset(name="bin",data=bin_)
        c_hist_bkg.create_dataset(name="count",data=count_)

    l_hist_bkg.create_dataset(name="bin",data=bin_)
    l_hist_bkg.create_dataset(name="count",data=count_)
    ###
    ### Signal histogram
    if not skip_flag:
        c_hist_sig = c_hist.create_group("signal")
    l_hist_sig = l_hist.create_group("signal")

    sig_hist_dict = dist_plot.signal_hist

    for sig_name in sig_hist_dict.keys():
        
        if not skip_flag:
            c_hist_sig_ = c_hist_sig.create_group(sig_name)# This is for each signal
        l_hist_sig_ = l_hist_sig.create_group(sig_name)# This is for each signal

        count_, bin_ = sig_hist_dict[sig_name]
        
        if not skip_flag:
            c_hist_sig_.create_dataset(name="bin",data=bin_)
            c_hist_sig_.create_dataset(name="count",data=count_)
        l_hist_sig_.create_dataset(name="bin",data=bin_)
        l_hist_sig_.create_dataset(name="count",data=count_)
        
    ###############################################################################
    
    # History saving
    if not skip_flag:
        c_history = complete.create_group("history")
    l_history = lite.create_group("history")
    
    for k in history_dict.keys():
        if not skip_flag:
            c_history.create_dataset(name=k,data=np.array(history_dict[k]))
        l_history.create_dataset(name=k,data=np.array(history_dict[k]))
    ###############################################################################
    if not skip_flag:
        complete.close()
    lite.close()
