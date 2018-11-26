import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

def histogram(img, l = 256):
    hist = np.zeros([l])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1
    return hist / (img.shape[0] * img.shape[1])

def equalizar_histograma(img, l = 256):
    
    #Calcula a funcao densidade de probabilidade - pdf
    pdf = histogram(img) #histograma da imagem
    
    #Calcula a funcao de distribuicao acumulada - cdf 
    cdf = np.zeros(l)
    for i in range(l):
        cdf[i] = np.sum(pdf[:i])
    cdf = cdf * (l - 1)
    cdf = cdf.round()
    
    #Remapeamento do histograma com base na cdf
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = cdf[img[i,j]]
    return img

def limiar(x):
    if x < 6 and x > 0:
        return 0
    else:
        return x  

def prepara_mascara(mascara):
    mascara = mascara[30:464, 2:637]
    mascara = mascara.astype(int)
    mascara[:179, 396:] = -255
    for i in range(mascara.shape[0]):
        mascara[i] = list(map(limiar, mascara[i]))
    return mascara

def hsv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #Converte RGB para HSV
    return image

def ycrcb(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) #Converte RGB para HSV
    return image

def lab(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) #Converte RGB para HSV
    return image

def carregar_imagem(filename, ext = '.jpg', color = 1):
    image = cv2.imread(filename + ext, color)
    return image

def convolucao(img, kernel):   
    ### add padding
    borda = (kernel.shape[0] - 1) // 2 
    img = np.vstack([np.zeros([borda, img.shape[1]]), img, np.zeros([borda, img.shape[1]])]) #adiciona borda em cima e embaixo da imagem com o valor de fundo
    img = np.hstack([np.zeros([img.shape[0], borda]), img, np.zeros([img.shape[0], borda])]) #adiciona borda aos lados da imagem com o valor de fundo
    img2 = np.zeros(img.shape)
    ###
    
    for i in range(borda, img.shape[0] - borda):
        for j in range(borda, img.shape[1] - borda):
            for u in range(kernel.shape[0]):
                for v in range(kernel.shape[1]):
                    img2[i, j] += img[i - borda + u, j - borda + v] * kernel[u, v]
                    
    img2 = img2[borda:-1 * borda, borda:-1 * borda] # remove padding

    ### normaliza valores de 0 a 255
    #img2 -= np.min(img2)
    #img2 = img2 / np.max(img2) * 255
    #img2 = np.uint(img2)
    ###
    return img2

def blurring(img):
    kernel = np.ones([5, 5])
    kernel = np.divide(kernel, 25)
    img = convolucao(img, kernel)
    return img

def high_pass(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    img = convolucao(img, kernel)
    return img

def add_layer(img, new_layer):
    new_array = np.zeros([img.shape[0], img.shape[1], img.shape[2] + 1])
    for k in range(img.shape[2]):
        new_array[:, :, k] = img[:, :, k]
    new_array[:, :, -1] = new_layer
    return new_array

def prepara_imagem(image):
    imageycrcb = ycrcb(image)
    blurred = blurring(imageycrcb[:, :, 0])
    image = add_layer(image, blurred)
    image = add_layer(image, array_x)
    image = add_layer(image, array_y)
    return image
    

def extrair_amostras(filename, color_map = ''):
    mascara = carregar_imagem(filename, color = 0)
    mascara = prepara_mascara(mascara)
    
    filename = filename[:-1]
    image = carregar_imagem(filename)
    image = image[30:464, 2:637, :]
    image = prepara_imagem(image)
    
    #if color_map == 'HSV':
    #    image = image_normalize_hsv(image)
        
    x = []
    y = []
    for i in range(mascara.shape[0]):
        for j in range(mascara.shape[1]):
            if mascara[i, j] > -1:
                x.append(image[i, j, :])
                if mascara[i, j] > 0:
                    y.append(0)
                else:
                    y.append(1)
    x= np.array(x)#.astype('uint8')
    y = np.array(y)#.astype('uint8')
    n_samples = x.shape[0]
    n_dimensions = x.shape[1]
    y_column = n_dimensions
    
    full_data = np.zeros([n_samples, n_dimensions + 1])
    full_data[:, :n_dimensions] = x
    full_data[:, y_column] = y
    
    full_data = pd.DataFrame(full_data)
    not_windshield = full_data[full_data[y_column] == 0]
    windshield = full_data[full_data[y_column] == 1]
    not_windshield = not_windshield.sample(n = int(1.5 * windshield.shape[0]))
    
    good_data = pd.concat([windshield, not_windshield], ignore_index = True)
    good_data = good_data.sample(frac = 1)
    
    X = good_data.values[:, 0:n_dimensions]
    Y = good_data.values[:, y_column]
    
    return X, Y

def train_classifier(clf, X_train, y_train):
    ''' Ajusta um classificador para os dados de treinamento. '''
    
    # Inicia o relógio, treina o classificador e, então, para o relógio
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Imprime os resultados
    print ("O modelo foi treinado em {:.4f} segundos".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Faz uma estimativa utilizando um classificador ajustado baseado na pontuação F1. '''
    
    # Inicia o relógio, faz estimativas e, então, o relógio para
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Imprime os resultados de retorno
    print ("As previsões foram feitas em {:.4f} segundos.".format(end - start))
    return f1_score(target, y_pred, pos_label=1)


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Treina e faz estimativas utilizando um classificador baseado na pontuação do F1. '''
    
    # Indica o tamanho do classificador e do conjunto de treinamento
    print ("Treinando um {} com {} pontos de treinamento. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Treina o classificador
    train_classifier(clf, X_train, y_train)
    
    # Imprime os resultados das estimativas de ambos treinamento e teste
    print ("Pontuação F1 para o conjunto de treino: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print ("Pontuação F1 para o conjunto de teste: {:.4f}.".format(predict_labels(clf, X_test, y_test)))

####################################################################
###                          EXECUÇÃO
####################################################################    

array_x = np.zeros([480, 640])
array_y = np.zeros([480, 640])
for i in range(480):
    for j in range(480):
        array_x[i, j] = i
        array_y[i, j] = j
array_x -= 480 // 2
array_y -= 640 // 2
array_x = array_x[30:464, 2:637]
array_y = array_y[30:464, 2:637]


n_dim = 6
X = np.zeros([1, n_dim])
Y = np.array([0])
dir_images = 'training_images' #diretorio que contem as imagens a serem analisadas
files = [] # Ler os arquivos no diretorio
for (dirpath, dirnames, filenames) in os.walk(os.getcwd() + '/' + dir_images):#'/home/jesse/ProcImg/Projeto03'):
    files.extend(filenames)
    break

for file in files:
    file = file[:-4]
    if file[-1] == 'm':
        new_X, new_Y = extrair_amostras(os.getcwd() + '/' + dir_images + '/' + file, 'RGB')
        X = np.concatenate((X, new_X), axis = 0)
        Y = np.concatenate((Y, new_Y))#, axis = 0)

X = X[1:,:]
Y = Y[1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 82)

clf1 = MLPClassifier(hidden_layer_sizes = (50, 3), activation = 'logistic')

train_predict(clf1, X_train, Y_train, X_test, Y_test)
