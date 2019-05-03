"""
Detecteur d intrusion avec python OpenCV et BackgroundSubtractorMOG

Il existe de  nombreuses façons d'effectuer la détection, le suivi et l'analyse de mouvement dans OpenCV. 
On a utilise dans ce programme:
Un modèle de mélange d'arrière-plan adaptatif amélioré par KaewTraKulPong et al., 
pour le suivi en temps réel avec détection des ombres 
disponible dans cv2.bgsegm.BackgroundSubtractorMOG
NB: lancez le programme en ligne de commande avec 3 parametres qui sont facultatif
1- -v chemin video pour les test sinon le programme utilise directement le webcam du pc
2- -s chemin d enregistrement du video de surveillance(le repertoire courant par default)
3- -i chemin d enregistrement du video d intruision(le repertoire courant par default)
lorsque vous lancez le programme dégagez le champs de vision de votre camera pendant au moins 20secondes 
pour que le  l algorithme modélise l arrière plan.
Gardez votre camera fixé pour une meilleur resultat
"""
import cv2
import numpy as np 
import imutils
from imutils.video import VideoStream
from threading import Thread
import playsound
import argparse
import time


def play_alarme(path):
    playsound.playsound(path)
    
arg=argparse.ArgumentParser()
arg.add_argument("-v", "--video", help="")
arg.add_argument("-s", "--surveillance", type=str, default="surveillance.avi", help="")
arg.add_argument("-i", "--intrusion", type=str, default="intrusion.avi", help="")
args=vars(arg.parse_args())

i=0
intrusion=15
alarme=False
video=None

bgsmog=cv2.bgsegm.createBackgroundSubtractorMOG()
if args.get("video", False):
    video=cv2.VideoCapture(args["video"])
else:
    video=VideoStream(src=0).start()
four_cc=cv2.VideoWriter_fourcc(*"MJPG")
writer=None
writer1=None


while True:
    img=video.read()
    img=img[1] if args.get("video", False) else img
    
    img=imutils.resize(img, width=650)
    blur=cv2.GaussianBlur(img, (17, 17), 0)
    gray=cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    erode=cv2.erode(gray, None, iterations=2)
    dilate=cv2.dilate(erode, None, iterations=2)
    bg=bgsmog.apply(dilate)
    seuil=cv2.threshold(bg, 117, 255, cv2.THRESH_BINARY)[1]
    cnts=cv2.findContours(seuil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    if len(cnts)>0:
        max_c=max(cnts, key=cv2.contourArea)
       
        if cv2.contourArea(max_c)>100:
            #box=cv2.minAreaRect(max_c)
            #box=cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            #box=np.array(box, dtype="int")
            #cv2.drawContours(img, [box], -1, (0, 0, 255), 2)
        
            (x, y, w, h)=cv2.boundingRect(max_c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            if writer is None:
                writer=cv2.VideoWriter(args["intrusion"], four_cc, 3, (img.shape[1], img.shape[0]), True)
            writer.write(img)    
            
            i+=1
            if i>=intrusion:
            
                cv2.putText(img, "ALERTE!!!!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)
                alarme=True
                if alarme:
                    t=Thread(target=play_alarme, args=("alarm.wav", ))
                    t.deamon=True
                    t.start() 
                    time.sleep(0.3)
                          
        else:
            i=0
            alarme=False 

    cv2.imshow("img", img)
    cv2.imshow("seuil", bg)
    if writer1 is None:
        writer1=cv2.VideoWriter(args["surveillance"], four_cc, 15, (img.shape[1], img.shape[0]), True)
    writer1.write(img)    
    key=cv2.waitKey(1) & 0xff
    if key==ord("q"):
        break

if args.get("video", False):
    video.release()
else:
    video.stop()   
cv2.destroyAllWindows()        

    

