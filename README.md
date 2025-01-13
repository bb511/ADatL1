# AXOL1TL Training Documentation

This repository contains the code and resources for setting up and running a comprehensive machine learning training pipeline. The pipeline is modular and easily customizable to accommodate various machine-learning tasks.

**The complete documentation can be found [here](https://codimd.web.cern.ch/ybDciz_5RR6wwA8wpdsgQA?view).**

Table of Contents
-----------------

*   [Dataset Creation and Preprocessing](#dataset-creation-and-preprocessing)
*   [Loss Function Setup](#loss-function-setup)
*   [Model Setup](#model-setup)
*   [Optimizer and Callbacks](#optimizer-and-callbacks)
*   [Score Managers and Plot Managers](#score-managers-and-plot-managers)
*   [Storage and Report Creation](#storage-and-report-creation)
*   [End-to-End Setup](#end-to-end-setup)

### Dataset Creation and Preprocessing

The dataset creation and preprocessing are handled within the `data_util` submodule of the `axo` module. The process is encapsulated in the `get_data` function in the `data.py` file, which orchestrates data reading, saturation treatment, normalization, quantization, and storage.

### Loss Function Setup

The `losses` submodule contains custom loss functions tailored for training. The core class is `L1ADBaseLoss`, which can be extended to create new loss functions. Examples include `cyl_PtPz_mae` and KLD loss.

### Model Setup

Model configurations are managed in the `models` submodule. The setup supports various AutoEncoder architectures, including Variational AutoEncoders (VAE). Model definitions, including encoder/decoder configurations and precision settings, are specified in the configuration file.

### Optimizer and Callbacks

Optimizers and callbacks are crucial for controlling the training process. Optimizers are configured similarly to TensorFlow/Keras, while callbacks manage tasks like learning rate adjustments. The `CosineDecayRestarts_Warmup` is a key callback used for dynamic learning rate scheduling.

### Score Managers and Plot Managers

Score management and visualization are handled by `Score Managers` and `Plot Managers`. These components manage thresholds and generate histograms for data distributions, aiding in model evaluation and visualization.

### Storage and Report Creation

The `store_axo` function manages the storage of training results, creating both comprehensive (`complete.h5`) and lightweight (`lite.h5`) files. Reports are generated using `generate_axolotl_html_report`, which creates detailed HTML and PDF reports documenting the training run.

### End-to-End Setup

The end-to-end setup integrates all components into a cohesive pipeline. Configurations can be passed as a dictionary or file path, allowing for flexible and customizable training runs. The setup combines the MASTER and SLAVE configurations to execute the entire pipeline.

#### How to Run the E2E Pipeline

##### Running from Command Line

**Default Config:**

    python3 master.py

**With Config File:**

    python3 master.py --config_path <path to config file>

##### Calling through Other Python Programs

**Default Configuration:**

    axo.main()  # Runs with default MASTER config.

**Custom Configuration Examples:**

    slave = {
        "model": {
            "encoder_config": {"nodes": [28, 64, 128]}
        }
    }
    axo.main(slave=slave)
