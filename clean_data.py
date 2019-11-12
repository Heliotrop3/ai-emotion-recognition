import csv
import pandas as pd

#Reads csv data files
dataTrain = pd.read_csv("Data\\faceexp-comparison-data-train-public.csv", error_bad_lines=False)
dataTest = pd.read_csv("Data\\faceexp-comparison-data-test-public.csv", error_bad_lines=False)

print(dataTrain.iloc[[0],[0,5,10]])