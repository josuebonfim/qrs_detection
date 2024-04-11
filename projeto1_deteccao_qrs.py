from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import firwin, iirdesign, lfilter, iirnotch, freqz, iirfilter, cheby1
import os
import pandas as pd

def deteccao_qrs(x, y):
    y = y[1: ] - y[:-1]
    M = np.max(np.abs(y))
    k = np.where(np.abs(y) > 0.6 * M)[0] # índices dos valores de X que ultrapassam o threshold
    sign_y = np.sign(y[k])
    mudanca_sinal_derivada = sign_y[1 :] - sign_y[:-1]
    k2 = np.where(mudanca_sinal_derivada == 2)[0] # índicies dos valores de x[k] que rolou a inversao de derivada
    k3 = np.where(mudanca_sinal_derivada == -2)[0]
    
    #Detecção do pico R
    for i in range (len(k3)):
        index = k[k3[i]]
        interval = x[index - 15 : index + 15]
        index_max = np.where(interval == np.max(interval))[0]
        k[k3[i]] = k[k3[i]] + np.argmax(interval) - 15
    picoR = k[k3]
    
    for i in range(len(picoR)):
        interval = x[picoR[i] - 20 : picoR[i] + 20]
        picoR_temp = np.argmax(interval) - 20
        picoR[i] += picoR_temp
    print(picoR)

    #deteccao do pico S e do pico Q
    picoS = []
    picoQ = []
    for i in range(len(picoR)):
        interval = x[picoR[i]: picoR[i] + 100]
        picoS.append(picoR[i] + np.argmin(interval))
        if(picoR[i] - 50 > 0):
            interval = x[picoR[i] - 50 : picoR[i]]
            picoQ.append(picoR[i] - 50 + np.argmin(interval))
        else:
            picoQ.append(0)
        
    return picoR, picoS, picoQ

def pre_process(x, fs = 500,freqs_notch = 188, freq_highpass = 1 , freq_lowpass = 245, freqs_qrs = [8, 12], duracao_minimia_deteccao_r = 0.025, tipo_filtro = 'cheby1', ordem_filtro = 2):
    y = x
    y = notch(y, ordem_filtro, freqs_notch, fs)
    plt.plot(np.abs(np.fft.fft(y))[:len(y)//2])
    plt.figure()
    plt.plot(y)
    plt.show()
    y = passa_baixas(y, ordem_filtro, freq_lowpass, fs)
    plt.plot(np.abs(np.fft.fft(y))[:len(y)//2])
    plt.figure()
    plt.plot(y)
    plt.show()
    # y = passa_altas(y, ordem_filtro, freq_highpass, fs)
    # plt.plot(np.abs(np.fft.fft(y))[:len(y)//2])
    # plt.figure()
    # plt.plot(y)
    # plt.show()
    y = passa_faixa(y, ordem_filtro, 10,fs)
    plt.plot(np.abs(np.fft.fft(y))[:len(y)//2])
    plt.figure()
    plt.plot(y)
    plt.show()
    return y

def notch(x, ordem_filtro, freq, fs):
    b, a = cheby1(ordem_filtro, 1, [freq-4, freq+4], btype="bandstop", fs = fs)
    y = lfilter(b, a, x)
    return y 

def passa_baixas(x, ordem_filtro, freq, fs):
    b, a = cheby1(ordem_filtro, 1, freq, btype="lowpass", fs = fs)
    y = lfilter(b, a, x)
    return y 

def passa_altas(x, ordem_filtro, freq, fs):
    b, a = cheby1(ordem_filtro, 1, freq, btype="highpass", fs = fs)
    y = lfilter(b, a, x)    
    return y 

def passa_faixa(x, ordem_filtro, freq, fs):
    banda_rejeicao = np.array([freq - 2, freq + 20])
    b, a = cheby1(ordem_filtro, 1, banda_rejeicao, btype="bandpass", fs = fs)
    y = lfilter(b, a, x)
    return y 


def feature_extraction(x, janela = 10 ,fs = 500,freqs_notch = [50, 150], freq_highpass = 0.6, freq_lowpass = 245, freqs_qrs = [8, 12]):
    #pre processamento
    y = pre_process(x)
    #janelamento
    features = []
    tamanho = int(janela * fs)
    for i in range (len(x)//tamanho):
        intervalo_x = x[i * tamanho: (i + 1) * tamanho]
        intervalo_y = y[i * tamanho: (i + 1) * tamanho]
        picoR, picoS, picoQ = deteccao_qrs(intervalo_x, intervalo_y)
        mediaRR = media_RR(intervalo_x, picoR)
        mediaRS = media_RS(intervalo_x, picoR, picoS)
        mediaQR = media_QR(intervalo_x, picoQ, picoR)
        razaoRS = np.mean(intervalo_x[picoS]/intervalo_x[picoR])
        razaoQR = np.mean(intervalo_x[picoQ]/intervalo_x[picoR])
        features.append([mediaRR, mediaRS, mediaQR, razaoRS, razaoQR])
    
    return np.array(features)

def media_ST(picoS, picoT):
    media = []
    for i in range(len(picoS)):
        media.append(picoT[i] - picoS[i])
    return np.mean(media)

def media_TP(picoT, picoP):
    media = []
    if len(picoT) > 1:
        for i in range(len(picoT) - 1):
            media.append(picoP[i + 1] - picoT[i])
    else:
        return 0
    return np.mean(media)
    
def media_RR(x, picoR):
    media = []
    if len(picoR) > 1:
        for i in range(len(picoR)-1):
            media.append(picoR[i + 1] - picoR[i])
    return np.mean(media)
    
def media_RS(x, picoR, picoS):
    media = []
    for i in range (len(picoR)):
        media.append(picoS[i] - picoR[i])
    return np.mean(media)

def media_QR(x, picoQ, picoR):
    media = []
    for i in range(len(picoR)):
        media.append(picoR[i] - picoQ[i])
    return np.mean(media)

def get_file_names(path):
    return os.listdir(path)

if __name__ == '__main__':
    features = []
    sinal_com_erro = []
    filename = "svt_evaluation_400_499.csv"
    path = "files/ECGData/"
    f = pd.read_csv(filename)
    files = f[f["Evaluation"] == 1]["FileName"].to_list()[0]
    
    # for i, file in enumerate(files[:1]) :
        # try:
    df = pd.read_csv(path + files + ".csv")
    signal = df.sum(axis = 1).to_numpy()
    feature = feature_extraction(signal)[0].round(2)
    features.append(feature)
        # except:
        #     print("deu rim")