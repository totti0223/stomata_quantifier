# bmicp (stomatal pore quantifier)

## Requirements
python>3
matplotlib==1.5.1
numpy==1.11.2
scipy==0.18.1
scikit_image==0.12.3
tensorflow==0.10.0rc0
Pillow==3.2.0
common==0.1.2
cv2==1.0
dlib==19.1.0
setuptools==32.3.1

## Installation

1. Download the repository.

2. Open terminal.

3. Move to the downloaded directory.

4. 'pip install .'

## Note

- Tensorflow must not be ver. 1.0.. Codes are not compatible.

- Several packages such as cv2 and dlib cannot be installed via pip in anaconda environment. In such cases, comment out the requirements.txt like the following 

	#cv2 ==1.0
	#dlib == 19.1.0

and install respectively via conda install ....

## Usage

- In terminal

'python'
'import bmicp'
'bmicp.cui("PATH/TO/THE/DIRECTORYORIMAGE")''

## Examples

TODO

