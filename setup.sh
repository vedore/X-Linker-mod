apt upgrade
apt update
apt install wget
apt install curl
apt install less
apt install nano
apt install unzip
apt install gawk
apt install libxml2-utils 
apt-get install xmlstarlet

# Install Pip requirements
pip install -r requirements.txt 


#--------------------------------------------------------------------------------
# Setup abbreviation detector
#--------------------------------------------------------------------------------
#Ab3P: https://github.com/ncbi-nlp/Ab3P

mkdir -p abbreviation_detector
cd abbreviation_detector/

#Get repositories
wget https://github.com/ncbi-nlp/Ab3P/archive/refs/heads/master.zip
unzip master.zip
mv Ab3P-master Ab3P

wget https://github.com/ncbi-nlp/NCBITextLib/archive/refs/heads/master.zip
unzip master.zip
mv NCBITextLib-master NCBITextLib

#Install 
apt-get install g++

# 1. Install NCBITextLib
cd NCBITextLib/lib/
make

cd ../../

## 2. Install Ab3P
cd Ab3P
sed -i "s#\*\* location of NCBITextLib \*\*#../NCBITextLib#" Makefile
sed -i "s#\*\* location of NCBITextLib \*\*#../../NCBITextLib#" lib/Makefile
make

cd ../

cd ../