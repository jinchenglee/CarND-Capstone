import xml.etree.ElementTree as ET
import os.path
import os
from glob import glob

# Go through dataset dir
# Rename file path of xml files

directory = "../data/dataset_xml/"

for f in glob(directory+'*.xml'):

    # Read 'off' class/category template
    tree = ET.parse(f)
    root = tree.getroot()

    filename = f[-14:-4]

    # Fill in folder/filename/path etc contents
    root.find('folder').text = "dataset_jpg"
    root.find('filename').text = filename+'.jpg'
    root.find('path').text = "dataset_jpg/"+filename+'.jpg'

    # Write updated contents out
    tree.write(f)


