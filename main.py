import streamlit as st

import numpy as np
import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import tempfile




st.set_page_config(page_title="Gender Recognition System",page_icon="https://gopostr.s3.amazonaws.com/favicon_url/3LS2j63vqtcDYL8I8MHIjYQuzkbZCmphFehSrtYM.png")


st.title("Gender Recognition System")
facemodel=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gendermodel=load_model("gendermodelfinal.h5")
classes=['women','man']
st.sidebar.image("https://www.legalindia.com/wp-content/uploads/2019/10/Gender-Detection-Test.jpg")
choice=st.sidebar.selectbox("My Menu",("Home","Image","Video","URL","Webcam"))

if(choice=="Home"):
    st.image("https://www.ceyond.group/wp-content/uploads/2021/03/Kopie-von-Ohne-Titel-3-1024x512.png")
    st.write("Gender Recognition System is a Computer vision deep learning application which can be used on detect the gender of a person in video as well as in the realtime Footage.")
elif(choice=="Image"):
    file=st.file_uploader("Upload an Image")
    if file:
        b=file.getvalue()                      #converting file into bytes
        k=np.frombuffer(b,np.uint8)           #encoding  image data into form of numpy array 
        img=cv2.imdecode(k,cv2.IMREAD_COLOR)   #decoding
        face=facemodel.detectMultiScale(img)
        for(x,y,l,w) in face:
            face_img=img[y:y+w,x:x+l]          #cropping the face
            cv2.imwrite('temp.jpg',face_img)           #saving the image into local directory        
            face_img=load_img('temp.jpg',target_size=(90,90,3))
            face_img=img_to_array(face_img)            #Converting the image into array
            face_img=np.expand_dims(face_img,axis=0)
           

            pred=gendermodel.predict(face_img)[0][0]
            if pred==1:
                     
                     
                     cv2.rectangle(img,(x,y),(x+l,y+w),(0,0,255),8)
                     cv2.putText(img,"Man",(x,y), cv2.FONT_HERSHEY_TRIPLEX,4, (0, 255, 0), 8)
            else:
                 
                 cv2.rectangle(img,(x,y),(x+l,y+w),(0,255,0),8)
                 cv2.putText(img,"Women",(x,y), cv2.FONT_HERSHEY_SIMPLEX,4, (0, 255, 0), 8)
        st.image(img,channels='BGR')      
            

elif(choice=='Video'):
    file=st.file_uploader('upload a video')
    window=st.empty()
    if file:
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)
        while (vid.isOpened()):     
            flag,frame=vid.read()
            if flag:
            
                
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face:
                    face_img=frame[y:y+w,x:x+l]  #cropping the face
                    cv2.imwrite('temp.jpg',face_img)  #Saving the image into local directory
                    face_img=load_img('temp.jpg',target_size=(90,90,3)) #load the image into a particular target size
                    face_img=img_to_array(face_img) #Converting the image into array
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=gendermodel.predict(face_img)[0][0]     
                    if pred==1:
                        
                        

                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                        cv2.putText(frame,"Man",(x,y), cv2.FONT_HERSHEY_TRIPLEX,2, (0, 255, 0), 2)
                    else:
                        


                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                        cv2.putText(frame,"Woman",(x,y), cv2.FONT_HERSHEY_TRIPLEX,2, (0, 255, 0), 2)

                window.image(frame,channels='BGR')
            

elif(choice=='URL'):
    
        
        link=st.text_input('Enter a link here:')
        btn=st.button('Start')
        window=st.empty()
        if btn:
        
            vid=cv2.VideoCapture(link)
        
            btn2=st.button("Stop")
            if btn2:
                vid.close()
                st.experimental_rerun()
            while(vid.isOpened()):
            
                flag,frame=vid.read()
                if flag:
                    face=facemodel.detectMultiScale(frame)
                    for (x,y,l,w) in face:
                        face_img=frame[y:y+w,x:x+l]  #cropping the face
                        cv2.imwrite('temp.jpg',face_img)  #Saving the image into local directory
                        face_img=load_img('temp.jpg',target_size=(90,90,3)) #load the image into a particular target size
                        face_img=img_to_array(face_img) #Converting the image into array
                        face_img=np.expand_dims(face_img,axis=0)
                        pred=gendermodel.predict(face_img)[0][0]     
                        if pred==1:
                        


                            cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                            cv2.putText(frame,"Man",(x,y), cv2.FONT_HERSHEY_TRIPLEX,2, (0, 255, 0), 2)
                        else:
                        


                            cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                            cv2.putText(frame,"Women",(x,y), cv2.FONT_HERSHEY_TRIPLEX,2, (0, 255, 0), 2)

                window.image(frame,channels='BGR')
            
    
            
        
elif(choice=="Webcam"):
    btn=st.button('Start')
    window=st.empty()
    if btn:
        
        vid=cv2.VideoCapture(0)
        btn2=st.button("Stop")
        if btn2:
            vid.close()
            st.experimental_rerun()
        while(vid.isOpened()):  
            flag,frame=vid.read()
            if flag:
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face:
                    face_img=frame[y:y+w,x:x+l]  #cropping the face
                    cv2.imwrite('temp.jpg',face_img)  #Saving the image into local directory
                    face_img=load_img('temp.jpg',target_size=(90,90,3)) #load the image into a particular target size
                    face_img=img_to_array(face_img) #Converting the image into array
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=gendermodel.predict(face_img)[0][0]     
                    if pred==1:
                        


                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                        cv2.putText(frame,"Man",(x,y), cv2.FONT_HERSHEY_TRIPLEX,2, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                        cv2.putText(frame,"Women",(x,y), cv2.FONT_HERSHEY_TRIPLEX,2, (255,0, 0), 2)


                        

            window.image(frame,channels='BGR')
    

       
        
        
