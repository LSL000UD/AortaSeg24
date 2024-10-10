chmod 777 /tmp/ \
&& apt-get update \
&& apt install libglib2.0-dev libgl1-mesa-glx -y \
&& pip install -r requirements.txt