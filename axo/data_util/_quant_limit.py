import numpy as np
import gc

def _minmax_qbit(bits: int, integer: int, keep_negative=True):
    """get range of a quantized_bits object. Symmetric is assumed to be False

    Returns:
        Tuple[float,float]: min,max
    """
    f = bits - integer
    return -2.**integer * keep_negative, 2.**integer - 2.**(keep_negative - f)

def _enforce_limit(data_bkg,data_sig,quantize_bits):
    l,h = _minmax_qbit(bits=quantize_bits[0],integer=quantize_bits[1])
    for event in data_bkg:
        data_bkg[event]["DATA"] = np.clip(data_bkg[event]["DATA"], l, h)
    for event in data_sig:
        data_sig[event]["DATA"] = np.clip(data_sig[event]["DATA"], l, h)
    return data_bkg,data_sig