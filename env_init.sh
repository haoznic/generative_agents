conda update --all 
conda init
>close
>open new 

conda create -n aiworld python=3.9.12 -y
conda activate aiworld

pip install ipykernel
python -m ipykernel install --user --name aiworld --display-name "py39(aiworld)"


# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# CUDA extension not installed.

# pip3 install torch torchvision torchaudio
# RuntimeError: probability tensor contains either `inf`, `nan` or element < 0

# pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA extension not installed.
# 不能匹配 auto-gptq

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install auto-gptq
# git clone https://github.com/PanQiWei/AutoGPTQ
# cd AutoGPTQ
# pip install .


pip install bitsandbytes --no-cache-dir
pip install "git+https://github.com/PanQiWei/AutoGPTQ.git@v0.4.2"

nvidia-smi
nvcc --version


pip install ipywidgets
# jupyter nbextension enable --py widgetsnbextension


python -u reverie.py>20240401-02.log 2>&1