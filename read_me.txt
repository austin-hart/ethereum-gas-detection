I found I had dependency issues when running on my machine so I ran this in a virtual environmentt.
This probably because I have anaconda installed and it has always created problems for me.
But what I did to make sure it works on any machine was create a virtual environment (found this on web):

pip install virtualenv

python3 -m venv env 

source env/bin/activate

//then install required dependencies

pip install --upgrade pip //need to be on latest version
pip install web3
pip install pandas
pip install matplotlib
pip install scipy
pip install scikit-learn

python gas_anomaly_detection.py


