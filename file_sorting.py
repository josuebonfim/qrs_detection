import os
import pandas as pd
import matplotlib.pyplot as plt 

if __name__ == '__main__':

    df = pd.read_csv("svt.csv")
    pacientes = df.FileName[400:500].to_list()
    path = "files/ECGData/"
    eval = []
    for paciente in pacientes:
        df_temp = pd.read_csv(path + paciente + ".csv")
        signal = df_temp.mean(axis = 1)
        plt.plot(signal)
        plt.show()
        if input("").upper() == "Y":
            eval.append(1)
        else:
            eval.append(0)