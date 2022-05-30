import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
import PIL 
import os 
import gdown
import tempfile
import shutil
from git import Repo

temp_dir = tempfile.TemporaryDirectory()
Data_preprocessing = tempfile.TemporaryDirectory(dir=temp_dir.name)
st.write('created temporary directory', Data_preprocessing.name)

st.title('Virtual Try-On')

os.chdir("./Virtual_Fashion_Try_On")

# Download Caffe Model for pose 
gdown.download('https://drive.google.com/uc?id=1hOHMFHEjhoJuLEQY0Ndurn5hfiA9mwko', "./pose/")

# Download lip model 
Repo.clone_from(url="https://github.com/levindabhi/Self-Correction-Human-Parsing-for-ACGPN.git", to_path="./Self-Correction-Human-Parsing-for-ACGPN/")
Repo.clone_from(url="https://github.com/levindabhi/U-2-Net.git", to_path="./U-2-Net/")

# download u-2 mask segmentation model 
url = 'https://drive.google.com/uc?id=16sEMJSR77HTic487suMU8fNP3lu57RLv'
output = 'lip_final.pth'
gdown.download(url, output, quiet=False)

os.mkdir("U-2-Net/saved_models/u2net")
os.mkdir("U-2-Net/saved_models/u2netp")
gdown.download("https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy", output="./U-2-Net/saved_models/u2netp/u2netp.pth")
gdown.download("https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ", output="./U-2-Net/saved_models/u2net/u2net.pth")
os.chdir("./U-2-Net")
print(os.cwd())
import u2net_load
import u2net_run
u2net = u2net_load.model(model_name = 'u2netp')
os.chdir("../")


# download pre-trained checkpoints 
os.mkdir("checkpoints")
gdown.download('https://drive.google.com/uc?id=1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx',output='checkpoints/ACGPN_checkpoints.zip', quiet=False)
os.chdir("./checkpoints/")
shutil.unpack_archive("ACGPN_checkpoints.zip", "./")
os.chdir("..")

clothes_File = st.file_uploader("Upload your cloth image below")
# st.write("The type is : ", type(clothes_File))
if clothes_File is not None:
	st.write(clothes_File.name)
	cloth_name = clothes_File.name
	clothes_File = Image.open(clothes_File)
	clothes_File = clothes_File.resize((192, 256), Image.BICUBIC).convert('RGB')

	test_color = tempfile.TemporaryDirectory(dir=Data_preprocessing.name)
	st.write('test color directory filepath: ', test_color.name)
	clothes_File_saved = os.path.join(test_color.name, cloth_name)
	clothes_File.save(clothes_File_saved)
	st.image(clothes_File_saved)

person_File = st.file_uploader("Upload your person image below")
if person_File is not None:
	st.write(person_File.name)
	person_name = person_File.name
	person_File = Image.open(person_File)
	person_File = person_File.resize((192,256), Image.BICUBIC)


	test_person = tempfile.TemporaryDirectory(dir=Data_preprocessing.name)
	person_File_saved = os.path.join(test_person.name, person_name)
	person_File.save(person_File_saved)
	st.image(person_File_saved)
#      # To read file as bytes:
#      bytes_data = uploaded_file.getvalue()
#      st.write(bytes_data)

#      # To convert to a string based IO:
#      stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#      st.write(stringio)

#      # To read file as string:
#      string_data = stringio.read()
#      st.write(string_data)

#      # Can be used wherever a "file-like" object is accepted:
#      dataframe = pd.read_csv(uploaded_file)
#      st.write(dataframe)