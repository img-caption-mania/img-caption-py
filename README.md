### img-caption-py

### clone repo
    
    git clone https://github.com/img-caption-mania/img-caption-py.git
    
    cd img-caption-py

### create env

    conda create -n myenv python=3.7
    
    conda activate myenv

### conda install requirements for notebook
    
    conda install notebook ipykernel nb_conda_kernels
    
    pip install -r requirements.txt
    
    or
    pip install -r requirements-gpu.txt # if u want to use GPU


### download dataset
    
    gdown https://drive.google.com/uc?id=1ZChja5DOLoeLMbJ-StFagQp5nRRGkWJn
    tar -xvzf image_gabung_with_feature.tar.gz
    
### run python script train n evaluate

    python image_captioning_good.py
