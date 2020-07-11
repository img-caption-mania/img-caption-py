### img-caption-py

### clone repo
    
    git clone https://github.com/img-caption-mania/img-caption-py.git
    
    cd img-caption-py

### create env

    conda create -n myenv python=3.6
    
    conda activate myenv

### conda install requirements for notebook
    
    conda install notebook ipykernel nb_conda_kernels
    
    pip install -r requirements.txt


### download dataset
    
    gdown https://drive.google.com/uc?id=1KPaAHJttW1P-CYoIf81mzFi8YRpSK_WH
    tar -xvzf image_gabung_with_feature.tar.gz
    
### run python script train n evaluate

    python image_captioning_good.py
