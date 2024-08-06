# RiverIceProject
Winter/Spring 24 Coop Project

Note that all files are in master branch.

## Training the Model

Start training the model by either:

1. Submitting `pygcn.sh` to the cluster:
   ```sh
   sbatch pygcn.sh

2. Requesting an interactive job and manually loading the modules specified in `pygcn.sh`, then running `main.py`:
    ```sh
    # Load the necessary modules as specified in pygcn.sh
    module load opencv python scipy-stack StdEnv/2023  gcc/12.3  gdal libspatialindex
    # Run the main script
    python main.py

## Script Details
`pygcn.sh`: This script includes several packages required to run main.py. There may be additional dependencies not included in the script that you will need to install manually.

The dependencies are the following:

      source ~/scratch/virenv_new/bin/activate
      module load opencv python scipy-stack StdEnv/2023  gcc/12.3  gdal libspatialindex
      
      pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
      pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
      pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
      pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
      pip install torch-geometric
      
      pip install requests
      pip install --no-index optuna
      pip install --no-index networkx
      pip install --no-index geopandas
        
      
      pip install --no-index shapely scikit_image scikit-learn imblearn


`constants.py`: This file contains several parameters that can be adjusted as needed. (maybe it should be better named something other than constants.py as it holds adjustable parameters rather than constants.)

## Data
The preprocessed data necessary for training the model is located in the data folder.

Raw data for preprocessing is not included in the repository, as it is not required for training the model.
