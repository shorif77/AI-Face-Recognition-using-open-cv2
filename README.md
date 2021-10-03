import cv2

!pip install face_recognition

import face_recognition

from google.colab import drive

drive.mount('/content/drive')

original_image = cv2.imread("/content/drive/MyDrive/image/image/image/friends.jpg")

shakil_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/shakil.jpg')
shakil_image_encoding = face_recognition.face_encodings(shakil_image)[0]

jewel_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/jewel.jpg')
jewel_image_encoding = face_recognition.face_encodings(jewel_image)[0]

abu_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/abu_bakar.jpg')
abu_image_encoding = face_recognition.face_encodings(abu_image)[0]

ananda_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/Ananda_m ozumder.jpg')
ananda_image_encoding = face_recognition.face_encodings(ananda_image)[0]

nozir_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/nozir.JPG')
nozir_image_encoding = face_recognition.face_encodings(nozir_image)[0]

jahangir_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/Jahangir.jpg')
jahangir_image_encoding = face_recognition.face_encodings(jahangir_image)[0]

farhad_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/farhad.jpg')
farhad_image_encoding = face_recognition.face_encodings(farhad_image)[0]

riyad_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/riyadjpg.jpg')
riyad_image_encoding = face_recognition.face_encodings(riyad_image)[0]

tarek_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/tarek.jpg')
tarek_image_encoding = face_recognition.face_encodings(tarek_image)[0]

sohid_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/sohid.JPG')
sohid_image_encoding = face_recognition.face_encodings(sohid_image)[0]

shorif_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/shorif.JPG')
shorif_image_encoding = face_recognition.face_encodings(shorif_image)[0]

rakib_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/rakib.JPG')
rakib_image_encoding = face_recognition.face_encodings(rakib_image)[0]

sourav_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/sourav.JPG')
sourav_image_encoding = face_recognition.face_encodings(sourav_image)[0]

farid_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/farid.jpg')
farid_image_encoding = face_recognition.face_encodings(farid_image)[0]

zobayer_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/zobayer.jpg')
zobayer_image_encoding = face_recognition.face_encodings(zobayer_image)[0]

yasin_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/yasin.jpg')
yasin_image_encoding = face_recognition.face_encodings(yasin_image)[0]

al_amin_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/al-amin.jpg')
al_amin_image_encoding = face_recognition.face_encodings(al_amin_image)[0]

Rohan_Ahmed_Sheikh_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/Rohan_Ahmed_Sheikh.jpg')
Rohan_Ahmed_Sheikh_image_encoding = face_recognition.face_encodings(Rohan_Ahmed_Sheikh_image)[0]

JH_Shuvo_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/JH_Shuvo.jpg')
JH_Shuvo_image_encoding = face_recognition.face_encodings(JH_Shuvo_image)[0]

Mahfuz_Al_Shams_Akash_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/Mahfuz Al Shams Akash.jpg')
Mahfuz_Al_Shams_Akash_image_encoding = face_recognition.face_encodings(Mahfuz_Al_Shams_Akash_image)[0]

Hossain_Nayan_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/Hossain Nayan.jpg')
Hossain_Nayan_image_encoding = face_recognition.face_encodings(Hossain_Nayan_image)[0]

Assaduzzaman_Pranto_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/Assaduzzaman Pranto.jpg')
Assaduzzaman_Pranto_image_encoding = face_recognition.face_encodings(Assaduzzaman_Pranto_image)[0]

abir_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/abir.jpg')
abir_image_encoding = face_recognition.face_encodings(abir_image)[0]

nazmul_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/naznul.jpg')
nazmul_image_encoding = face_recognition.face_encodings(nazmul_image)[0]

main_uddin_image = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/main_uddin.jpg')
main_uddin_image_encoding = face_recognition.face_encodings(nazmul_image)[0]

known_face_encodings =[ shakil_image_encoding, jewel_image_encoding, abu_image_encoding, 
                       ananda_image_encoding, nozir_image_encoding, jahangir_image_encoding,farhad_image_encoding,
                       riyad_image_encoding,tarek_image_encoding,sohid_image_encoding,shorif_image_encoding,
                       rakib_image_encoding,sourav_image_encoding, farid_image_encoding,zobayer_image_encoding, yasin_image_encoding,
                       al_amin_image_encoding, Rohan_Ahmed_Sheikh_image_encoding, JH_Shuvo_image_encoding, Mahfuz_Al_Shams_Akash_image_encoding,
                       Hossain_Nayan_image_encoding,Assaduzzaman_Pranto_image_encoding,abir_image_encoding, nazmul_image_encoding ,main_uddin_image_encoding  ]
known_face_names = ['shakil', 'jewel','abu','ananda','nozir','jahangir','farhad','riyad','tarek','sohid',
                    'shorif','rakib','sourav','farid','zobayer' , 'yasin','al_amin','Rohan_Ahmed_Sheikh',
                    'JH_Shuvo','Mahfuz_Al_Shams_Akash','Hossain_Nayan','Assaduzzaman_Pranto','abir', 'nazmul','main_uddin']

face_to_recognize = face_recognition.load_image_file('/content/drive/MyDrive/image/image/image/friends.jpg')
face_locations = face_recognition.face_locations(face_to_recognize, model='hog')
face_encodings = face_recognition.face_encodings(face_to_recognize,face_locations)

from google.colab.patches import cv2_imshow

for current_face_location, current_face_encoding in zip(face_locations, face_encodings):
  top_pos, right_pos, bottom_pos, left_pos = current_face_location
  all_matches = face_recognition.compare_faces(known_face_encodings,current_face_encoding)
  name_of_person = 'Unrecognized'
  if True in all_matches:
    matched_index = all_matches.index(True)
    name_of_person = known_face_names[matched_index]
  cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
  font = cv2.FONT_HERSHEY_DUPLEX
  cv2.putText(original_image,name_of_person,(left_pos,top_pos-10),font,0.6,(0,255,0))

cv2_imshow(original_image)
