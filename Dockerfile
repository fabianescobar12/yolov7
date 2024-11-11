FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel 

#RUN virtualenv /env -p python3

RUN apt-get update -y \ 
    && apt-get install -y git 
    

RUN git clone https://github.com/SrCarlos01/yolov7.git

#RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /yolov7