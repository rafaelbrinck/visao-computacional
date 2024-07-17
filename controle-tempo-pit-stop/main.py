import cv2
import numpy as np
import datetime

# Definição das regiões de interesse (ROI)
VAGAS = [
    [145, 108, 352, 154]
]

# Constantes
LIMITE_VAGA_LIVRE = 18900
LIMITE_VAGA_OCUPADA = 20000
NUM_VAGAS = len(VAGAS)
DELAY = 50


tempo = {i: {"entrada": None, "saida": None, "final": 0} for i in range(NUM_VAGAS)}

def processa_frame(img):
    """
    Processa a imagem para destacar as áreas de interesse.
    """
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_threshold = cv2.adaptiveThreshold(img_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    img_blur = cv2.medianBlur(img_threshold, 5)
    kernel = np.ones((3, 3), np.int8)
    img_dil = cv2.dilate(img_blur, kernel)
    return [img_dil, img_cinza]

def verifica_vagas(img, img_dil, vagas, fps, frame):
    """
    Verifica o status das vagas de estacionamento e desenha as regiões na imagem.
    """
    qt_vagas_abertas = 0
    for i, (x, y, w, h) in enumerate(vagas):
        recorte = img_dil[y:y+h, x:x+w]
        qt_px_branco = cv2.countNonZero(recorte)

        cv2.rectangle(img, (x, y+h-22), (x+50, y+h-5), (0, 0, 0), -1)
        cv2.putText(img, str(qt_px_branco), (x, y+h-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        if qt_px_branco > LIMITE_VAGA_OCUPADA:
            cor = (0, 0, 255)  # Vermelho
            if tempo[i]["entrada"] is None:
                tempo[i]["entrada"] = frame
        elif LIMITE_VAGA_LIVRE < qt_px_branco <= LIMITE_VAGA_OCUPADA:
            cor = (0, 255, 255)  # Amarelo
        else:
            cor = (0, 255, 0)  # Verde
            qt_vagas_abertas += 1
            if tempo[i]["entrada"] is not None:
                tempo[i]["saida"] = frame
                total = tempo[i]["saida"] - tempo[i]["entrada"]
                tempo[i]["final"] = total / fps
                tempo[i]["entrada"] = None
                tempo[i]["saida"] = None
                

        cv2.rectangle(img, (x, y), (x + w, y + h), cor, 2)
    
    return qt_vagas_abertas

def exibe_status(img, qt_vagas_abertas, num_vagas):
    """
    Exibe o status das vagas abertas na imagem.
    """

    for i in range(NUM_VAGAS):
        if tempo[i]["final"] > 0:
            mensagem = f"Tempo: {tempo[i]["final"]:.3f} seg."
            cv2.rectangle(img, (90, 0), (555, 60), (0, 0, 0), -1)
            cv2.putText(img, mensagem, (100, 45), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 5)


def main():
    video_path = 'controle-tempo-pit-stop/f1-car.mp4'
    video = cv2.VideoCapture(video_path)

    # Fazendo a coleta de frames e fps do vídeo e transformando em segundos para coletar a informação exata de tempo
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    segundos = int(frames / fps) 
    tempo_total = str(datetime.timedelta(seconds=segundos)) 


    print("Tempo total do video:", tempo_total) 

    if not video.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return
    
    frame = 0

    while True:
        check, img = video.read()
        if not check:
            break

        img_dil = processa_frame(img)
        qt_vagas_abertas = verifica_vagas(img, img_dil[0], VAGAS, fps, frame)
        exibe_status(img, qt_vagas_abertas, NUM_VAGAS)

        cv2.imshow('Video', img)
        frame += 1
        


        if cv2.waitKey(DELAY) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
