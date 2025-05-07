1.Model establishment and training

Model: deeplabv3

Data: The training dataset is built by Medetec_foot_ulcer_224 
  
install Requirements
    pip install -r requirements.txt
    
Run
    python train.py
    python predictout.py
  

Run Front-End：
cd ./front-end/html-frontend
http-server -p 8080

Run Back-End：
python back-end/api.py