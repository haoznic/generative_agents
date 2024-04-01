conda update --all 
conda init
>close
>open new 

conda create -n aiworld python=3.9.12 -y
conda activate aiworld

pip install ipykernel
python -m ipykernel install --user --name aiworld --display-name "py39(aiworld)"


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pip install bitsandbytes --no-cache-dir
# pip install "git+https://github.com/PanQiWei/AutoGPTQ.git@v0.4.2"

nvidia-smi
nvcc --version


pip install ipywidgets
# jupyter nbextension enable --py widgetsnbextension

# run backend_server/reverie.py 
# -u std out log without buffer
python -u reverie.py>20240401-02.log 2>&1