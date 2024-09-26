# Inicializar python enviroment para jupyter notebook no vscode
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
python3 -m pip install jupyter ipykernel
```
```bash
python3 -m ipykernel install --user --name=jupyter --display-name "Python (.jupyter)"
```
```bash
pip3 install -r requirements.txt
```