import xml.etree.ElementTree as ET
import os.path

# Read 'off' class/category template
tree = ET.parse('data/off_template.xml')
root = tree.getroot()

# Go through each dir
# If label file exits, print out labeled contents
# If not, add a "off" category of label xml file

# Dir 1
directory = "data/just_traffic_out/"
for num in range(712):
    filename = "left"+str(num).zfill(4)
    if os.path.isfile(directory+filename+'.xml'):
        #print directory+filename+'.xml'
        tree2 = ET.parse(directory+filename+'.xml')
        root2 = tree2.getroot()
        print directory+filename+'.xml', root2.find('object').find('name').text
    else:
        # Fill in folder/filename/path etc contents
        root.find('folder').text = "just_traffic"
        root.find('filename').text = filename+'.jpg'
        root.find('path').text = "/work/git_repo/CarND_Zoom_Ahead/data/just_traffic/"+filename+'.jpg'

        # Write updated contents out
        tree.write(directory+filename+'.xml')

# Dir 2
directory = "data/loop_traffic_out/"
for num in range(1151):
    filename = "left"+str(num).zfill(4)
    if os.path.isfile(directory+filename+'.xml'):
        #print directory+filename+'.xml'
        tree2 = ET.parse(directory+filename+'.xml')
        root2 = tree2.getroot()
        print directory+filename+'.xml', root2.find('object').find('name').text
    else:
        # Fill in folder/filename/path etc contents
        root.find('folder').text = "loop_traffic"
        root.find('filename').text = filename+'.jpg'
        root.find('path').text = "/work/git_repo/CarND_Zoom_Ahead/data/loop_traffic/"+filename+'.jpg'

        # Write updated contents out
        tree.write(directory+filename+'.xml')

