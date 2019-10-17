
mkdir weights
git clone https://github.com/kayoyin/antialiased-cnns.git blur
pip -r install blur/requirements.txt
sh blur/weights/download_antialiased_models.sh
cd weights
wget https://download.pytorch.org/models/resnet18-5c106cde.pth

cd ..
split_folders data/train --output datasplit --ratio .8 .2