from fastai import *
from fastai.text import *
from fastai.core import *
import tensorflow
import torch
import opencv
import csv


path = 'C:\\Users\\%USERNAME\\Documents\\GitHub\\ai-emotion-recognition\\Data'

dataLearn = TextLMDataBunch.from_csv(Path(path), 'faceexp-comparison-data-test-public.csv')