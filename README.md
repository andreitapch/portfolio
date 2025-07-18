To install the environment go to 
cd .\fraud_detection\
Then 
python -m venv venv

to activate 
venv\Scripts\activate
python.exe -m pip install --upgrade pip

#first time users : 
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn `
transformers jupyterlab seaborn matplotlib pytest pytest-dash black flake8

##pip freeze > requirements.txt

#to reproduce this code just install this 
pip install -r requirements.txt


python -m ipykernel install --user --name=fraud-detection
