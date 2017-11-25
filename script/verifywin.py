#!/usr/bin/env python
from lxml import etree
from io import StringIO
import cv2
import os
from glob import glob
from winrealworld_classifier import RealWorldClassifier 
#from light_classification.wintl_classifier import TLClassifier

resultmap = {1.0:'green', 2.0:'red', 3.0:'yellow', 4.0:'off'}

class Verify(object):
    def process(self, imagedir, labeldir):
        #issimulator = False
        #classifier = TLClassifier(issimulator)
        classifier = RealWorldClassifier()
        correctcount = 0
        incorrectcount = 0
        counter = 0
        for imagepath in glob(os.path.join(imagedir, '*.jpg')):
            img = cv2.imread(imagepath)
            idx = imagepath.rfind(os.path.sep)
            name = imagepath[idx+1:] if idx >= 0 else imagepath
            name = '{}{}'.format(name[:-3], 'xml')
            labelpath = os.path.join(labeldir, name)

            label = 'no traffic light'
            if os.path.isfile(labelpath):
		        #label exists
                tree = etree.parse(labelpath)

                root = tree.getroot()
                namelist = root.findall('object/name')
                for name in namelist:
                    text = name.text
                    if text == 'red':
                        label = 'red'
                    elif text == 'green':
                        label = 'green'
                    elif text == 'yellow':
                        label = 'yellow'
                    elif text == 'off':
                        label = 'off'
                    else:
                        print ('label not recognize: ',text)
            result, prob = classifier.get_classification(img)
            mylabel = resultmap.get(result, 'unknown')
            if mylabel == label:
                correctcount += 1
            else:
                incorrectcount += 1
                print (imagepath, label, 'vs', mylabel, '(result=', result, ',prob=', prob)
            counter += 1
        print ('correct count ', correctcount, 'incorrect count ', incorrectcount, 100.0*correctcount/(correctcount + incorrectcount))


if __name__ == '__main__':
    try:
        v = Verify()
	    #print 'testing image from simulator'
        print ('testing image from real world')
        v.process('/work/git_repo/CarND-Capstone/data/eval_jpg',
                    '/work/git_repo/CarND-Capstone/data/eval_xml')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print ('catch exception ' + str(e))
