from dataclasses import dataclass
import numpy as np
import gc
import os
import re
import h5py

@dataclass
class L1DataProcessor:
    
    processed_data_path: str
    constituents: dict
    saturation_mode: str
    norm_scheme: str
    norm_ignore_zeros: bool
    quantize_bits: tuple

    def process(self, background_datadict: dict, signal_datadict: dict):
        """Applies processing to the data and stores result in a file."""

        # Remove saturation
        background_datadict = self.remove_saturation(background_datadict)
        signal_datadict = self.remove_saturation(signal_datadict)

        # Normalization
        background_datadict, signal_datadict = self.normalize(background_datadict, signal_datadict)

        # Quantization
        background_datadict, signal_datadict = self.quantize(background_datadict, signal_datadict)

        # Store the processed data in a file
        self.pack(background_datadict, signal_datadict)


    def remove_saturation(self, datadict: dict):
        ET_thres = 4095  # saturated ET entries will always be dropped
        nof_MET = np.sum(self.constituents['MET'])
        nof_egamma = np.sum(self.constituents['EGAMMA'])
        nof_mu = np.sum(self.constituents['MUON'])
        nof_jet = np.sum(self.constituents['JET'])

        pt_thres = np.array([2047.5] * nof_MET + [255.5] * (nof_egamma + nof_mu) + [1023.5] * nof_jet) * 2
        
        # ET saturation treatment ...
        for event in datadict.keys():
            mask = datadict[event]["META"]["ET"] < ET_thres
            datadict[event]["DATA"] = datadict[event]["DATA"][mask]
            for k in datadict[event]["META"].keys():
                datadict[event]["META"][k] = datadict[event]["META"][k][mask]
                
        # pT saturation treatment ....
        if self.saturation_mode == "drop":
            for event in datadict.keys(): # In case of BKG the events are Train and Test
                mask = np.all(datadict[event]["DATA"][:,:,0] < pt_thres, axis=1)
                datadict[event]["DATA"] = datadict[event]["DATA"][mask]

                for k in datadict[event]["META"].keys():
                    datadict[event]["META"][k] = datadict[event]["META"][k][mask]
        else:
            for event in datadict.keys(): # In case of BKG the events are Train and Test
                mask = datadict[event]["DATA"][:,:,0] < pt_thres
                datadict[event]["DATA"] *= mask[:, :, None]
        del mask
        gc.collect()
        return datadict


    def normalize(self, background_datadict: dict, signal_datadict: dict):
        train_fea = background_datadict["Train"]["DATA"]

        train_fea = train_fea.astype(np.float32)
        norm_scale = np.ones(train_fea.shape[1:])  # (#constituents, #vec)
        norm_bias = np.zeros(train_fea.shape[1:])

        match = re.findall(r'RobustScaler(?:_pow2|)\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\)', self.norm_scheme)

        percentiles = float(match[0][0]), float(match[0][1])
        l_1, h_1 = float(match[0][2]), float(match[0][3])

        mask = np.ones(train_fea.shape[0], dtype=np.bool_)
        for i in range(train_fea.shape[1]):
            if self.norm_ignore_zeros:
                mask = train_fea[:, i, 0] != 0  # pt != 0
            l_0, h_0 = np.percentile(train_fea[:, i][mask], percentiles, axis=0)
            norm_scale[i] = (h_0 - l_0) / (h_1 - l_1)
            norm_bias[i] = (l_0 * h_1 - h_0 * l_1) / (h_1 - l_1)

        norm_scale += norm_scale == 0  # avoid division by zero

        if self.norm_scheme.startswith('RobustScaler_pow2'):
            norm_scale = np.power(2, np.ceil(np.log2(norm_scale)))
            norm_bias = np.round(norm_bias)

        ### Applying the normalisation to all the data

        for event in background_datadict:
            background_datadict[event]["DATA"] = (background_datadict[event]["DATA"] - norm_bias)/(norm_scale)
        for event in signal_datadict:
            signal_datadict[event]["DATA"] = (signal_datadict[event]["DATA"] - norm_bias)/(norm_scale)

        del train_fea
        gc.collect()

        self.norm_bias, self.norm_scale = norm_bias, norm_scale
        return background_datadict, signal_datadict
    
    
    @staticmethod
    def _minmax_qbit(bits: int, integer: int, keep_negative=True):
        """get range of a quantized_bits object. Symmetric is assumed to be False

        Returns:
            Tuple[float,float]: min,max
        """
        f = bits - integer
        return -2.**integer * keep_negative, 2.**integer - 2.**(keep_negative - f)

    def quantize(self, background_datadict: dict, signal_datadict: dict):
        l,h = self._minmax_qbit(
            bits=self.quantize_bits[0],
            integer=self.quantize_bits[1]
        )
        for event in background_datadict:
            background_datadict[event]["DATA"] = np.clip(background_datadict[event]["DATA"], l, h)
        for event in signal_datadict:
            signal_datadict[event]["DATA"] = np.clip(signal_datadict[event]["DATA"], l, h)
        return background_datadict, signal_datadict
    

    def pack(
        self,
        background_datadict: dict,
        signal_datadict: dict,
    ):
        """Store the processed data in a file."""
        
        if os.path.exists(self.processed_data_path):
            os.remove(self.processed_data_path)
        
        f = h5py.File(self.processed_data_path,"w")
        
        # Adding the configs ....
        # config = json.dumps(config)
        # f.attrs["config"] = config
        # Filling the Background data
        
        f.create_group("Background_data")
        for event in background_datadict.keys():
            f["Background_data"].create_group(event)
            # Filling the Meta data first
            for meta_key in background_datadict[event]["META"].keys():
                f["Background_data"][event].create_dataset(name=meta_key,
                                                        data=background_datadict[event]["META"][meta_key],
                                                        # compression="gzip",  ## Uncomment to reduce file size
                                                        # compression_opts=9 ## Uncomment to reduce file size
                                                        )
            # Filling the data
            f["Background_data"][event].create_dataset(name="DATA",
                                                    data=background_datadict[event]["DATA"],
                                                    # compression="gzip", ## Uncomment to reduce file size
                                                    # compression_opts=9 ## Uncomment to reduce file size
                                                    )
        # Filling the signal data
        f.create_group("Signal_data")
        for event in signal_datadict.keys():
            f["Signal_data"].create_group(event)
            # Filling the Meta data first
            for meta_key in signal_datadict[event]["META"].keys():
                f["Signal_data"][event].create_dataset(name=meta_key,
                                                    data=signal_datadict[event]["META"][meta_key],
                                                    # compression="gzip", ## Uncomment to reduce file size
                                                    # compression_opts=9 ## Uncomment to reduce file size
                                                    )
            # Filling the data
            f["Signal_data"][event].create_dataset(name="DATA",
                                                data=signal_datadict[event]["DATA"],
                                                # compression="gzip", ## Uncomment to reduce file size
                                                # compression_opts=9 ## Uncomment to reduce file size
                                                )
        # Filling the norms and biases
        f.create_group("Normalisation")
        f["Normalisation"].create_dataset(
            name = "norm_scale",
            data = self.norm_scale,
            # compression="gzip", ## Uncomment to reduce file size
            # compression_opts=9 ## Uncomment to reduce file size
        )
        
        f["Normalisation"].create_dataset(
            name = "norm_bias",
            data = self.norm_bias,
            # compression="gzip", ## Uncomment to reduce file size
            # compression_opts=9 ## Uncomment to reduce file size
        )

        f.close()
        gc.collect()