from . import _normalise
from . import _pack
from . import _quant_limit
from . import _read
from . import _saturation
import gc

def get_data(config_master):
    # This config is supposed to be the data config for the file
    
    ### Reading the data
    # <---- This is where the configs will be later modified to remove redundancy :)
    config_bkg = config_master["Read_configs"]["BACKGROUND"]
    config_sig = config_master["Read_configs"]["SIGNAL"]
    
    data_bkg = _read.get_data_background(config=config_bkg)
    data_sig = _read.get_data_signal(config=config_sig)
    #############################################################################
    ### Saturation Treatment
    saturation_mode = config_master["Saturation_configs"]["saturation_mode"]

    data_bkg = _saturation._remove_saturation(data_bkg,
                                              config_bkg["constituents"],
                                              saturation_mode=saturation_mode)
    data_sig = _saturation._remove_saturation(data_sig,
                                              config_sig["constituents"],
                                              saturation_mode=saturation_mode)
    #############################################################################
    ### Normalisation
    scheme = config_master["Normalisation_configs"]["scheme"]
    norm_ignore_zeros = config_master["Normalisation_configs"]["norm_ignore_zeros"]

    data_bkg, data_sig, norm_bias, norm_scale = _normalise(data_bkg = data_bkg,
                                                           data_sig = data_sig,
                                                           scheme = scheme,
                                                           norm_ignore_zeros = norm_ignore_zeros)
    #############################################################################
    ### Quantisation
    
    quantize_bits = config_master["Quantization_configs"]["quantize_bits"]

    data_bkg, data_sig = _quant_limit._enforce_limit(data_bkg = data_bkg,
                                        data_sig = data_sig,
                                        quantize_bits = quantize_bits)
    #############################################################################
    ### Packing and storing
    _pack(data_bkg = data_bkg,
                data_sig=data_sig,
                config=config_master,
                norm_bias=norm_bias,
                norm_scale=norm_scale
               )
    
    
    del data_bkg, data_sig
    gc.collect()