# handwritten-character-recognition

This a Deep learning AI system which recognize handwritten characters, Here I use chars74k data-set for training the model

### Prerequisites:

- Python
- anaconda
- Pip
- virtualenv

Download handwritten dataset from [here](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz)

It has only 55 samples for each class, so I have written script to create duplicate images with different backgroud color. 

Clone this repository and create a virtualenv using below command
```
virtualenv venv
source venv/bin/activate
```
Navigate to cloned directory
```
pip install -r requirements.txt
```

Create duplicate images for dataset
```
python generate_dataset.py
```

Open notebook
```
jupyter notebook
```

