import numpy as np
import face_recognition as fr
import cv2
from engine import get_rostos

rostos_conhecidos, nomes_dos_rostos = get_rostos()

video_capture = cv2.VideoCapture(0)
while True: 
    ret, frame = video_capture.read()

    if not ret:
        print("Falha ao capturar a imagem da webcam.")
        break

    # Reduzir o tamanho do quadro capturado para processá-lo mais rapidamente
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Encontre todas as localizações de rostos na imagem atual do quadro de vídeo
    localizacao_dos_rostos = fr.face_locations(rgb_small_frame)

    if not localizacao_dos_rostos:
        print("Nenhum rosto encontrado.")
        continue

    # Obter encodings dos rostos encontrados
    try:
        rosto_desconhecidos = fr.face_encodings(rgb_small_frame, localizacao_dos_rostos)
    except Exception as e:
        print(f"Erro ao calcular os encodings dos rostos: {e}")
        continue

    for (top, right, bottom, left), rosto_desconhecido in zip(localizacao_dos_rostos, rosto_desconhecidos):
        resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)
        face_distances = fr.face_distance(rostos_conhecidos, rosto_desconhecido)

        melhor_id = np.argmin(face_distances)
        if resultados[melhor_id]:
            nome = nomes_dos_rostos[melhor_id]
        else:
            nome = "Desconhecido"

        # Redimensionar as coordenadas do rosto de volta ao tamanho original do quadro
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenhe um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Desenhe um retângulo com o nome embaixo do rosto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Exiba a imagem resultante
    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
