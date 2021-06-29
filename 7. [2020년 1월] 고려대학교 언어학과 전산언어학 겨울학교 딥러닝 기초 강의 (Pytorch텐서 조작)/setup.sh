# 

cd /content

echo "Installing openjdk-8-jdk...."
! sudo apt-get install openjdk-8-jdk
echo "Done"

echo "Installing Spacy"
! pip install spacy
! python -m spacy download en
echo "Done"

echo "Installing nltk"
! pip install nltk
echo "Done"

echo "Installing konlpy....."
! pip3 install konlpy
echo "Done"

echo "Installing JPype1-py3...."
! pip3 install JPype1-py3
echo"Done"

echo "Installing mecab-0.996-ko-0.9.2.tar.gz....."

echo "Downloading mecab-0.996-ko-0.9.2.tar.gz......."
echo "from https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz"
! wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz 
echo "Done"

echo "Unpacking mecab-0.996-ko-0.9.2.tar.gz......."
! tar xvfz mecab-0.996-ko-0.9.2.tar.gz > /dev/null 2>&1 
echo "Done"

echo "Change Directory to mecab-0.996-ko-0.9.2......."
cd mecab-0.996-ko-0.9.2/

echo "installing mecab-0.996-ko-0.9.2.tar.gz........"
echo 'configure'
! ./configure > /dev/null 2>&1
echo 'make'
! make > /dev/null 2>&1
echo 'make check'
! make check > /dev/null 2>&1
echo 'make install'
! make install > /dev/null 2>&1

echo 'ldconfig'
! ldconfig > /dev/null 2>&1
echo "Done"

echo "Change Directory to /content"
cd ../

echo "Downloading mecab-ko-dic-2.1.1-20180720.tar.gz......."
echo "from https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz"
! wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
echo "Done"
 
echo "Unpacking  mecab-ko-dic-2.1.1-20180720.tar.gz......."
! tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz > /dev/null 2>&1
echo "Done"

echo "Change Directory to mecab-ko-dic-2.1.1-20180720"
cd mecab-ko-dic-2.1.1-20180720/
echo "Done"

echo "installing........"
echo 'configure'
! ./configure > /dev/null 2>&1
echo 'make'
! make > /dev/null 2>&1
echo 'make install'
! make install > /dev/null 2>&1

echo 'apt install curl'
! apt install curl > /dev/null 2>&1
echo 'apt install git'
! apt install git > /dev/null 2>&1
echo 'bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)'
! bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)  > /dev/null 2>&1
echo "Done"

echo "Getting nsmc contents"
! cd /content
! git clone https://github.com/e9t/nsmc.git
echo "Done"

echo "Setup Finished!"
