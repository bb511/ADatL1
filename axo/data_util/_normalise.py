import numpy as np
import gc
import re

def _normalise(data_bkg,data_sig,scheme,norm_ignore_zeros):
    train_fea = data_bkg["Train"]["DATA"]

    train_fea = train_fea.astype(np.float32)
    norm_scale = np.ones(train_fea.shape[1:])  # (#constituents, #vec)
    norm_bias = np.zeros(train_fea.shape[1:])

    match = re.findall(r'RobustScaler(?:_pow2|)\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\)', scheme)

    percentiles = float(match[0][0]), float(match[0][1])
    l_1, h_1 = float(match[0][2]), float(match[0][3])

    mask = np.ones(train_fea.shape[0], dtype=np.bool_)
    for i in range(train_fea.shape[1]):
        if norm_ignore_zeros:
            mask = train_fea[:, i, 0] != 0  # pt != 0
        l_0, h_0 = np.percentile(train_fea[:, i][mask], percentiles, axis=0)
        norm_scale[i] = (h_0 - l_0) / (h_1 - l_1)
        norm_bias[i] = (l_0 * h_1 - h_0 * l_1) / (h_1 - l_1)

    norm_scale += norm_scale == 0  # avoid division by zero

    if scheme.startswith('RobustScaler_pow2'):
        norm_scale = np.power(2, np.ceil(np.log2(norm_scale)))
        norm_bias: np.ndarray = np.round(norm_bias)

    ### Applying the normalisation to all the data

    for event in data_bkg:
        data_bkg[event]["DATA"] = (data_bkg[event]["DATA"] - norm_bias)/(norm_scale)
    for event in data_sig:
        data_sig[event]["DATA"] = (data_sig[event]["DATA"] - norm_bias)/(norm_scale)

    del train_fea
    gc.collect()
    return data_bkg, data_sig, norm_bias, norm_scale