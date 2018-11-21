import cv2
#from matplotlib import pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os


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
    mascara[:179, 396:] = -1
    for i in range(mascara.shape[0]):
        mascara[i] = list(map(limiar, mascara[i]))
    return mascara

def image_normalize_hsv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #Converte RGB para HSV
    image[:,:,2] = equalizar_histograma(image[:,:,2])
    #image = image.astype('float')
    #image[:, :, 0] = image[:, :, 0] / 360
    #image[:, :, 1] = image[:, :, 1] / 100
    #image[:, :, 2] = image[:, :, 2] / np.average(image[:, :, 2])
    return image


def extrair_amostras(filename, color_map = ''):
    mascara = cv2.imread(filename + '.jpg', 0)
    mascara = prepara_mascara(mascara)
    
    filename = filename[:-1]
    image = cv2.imread(filename + '.jpg', 1)
    image = image[30:464, 2:637, :]
    if color_map == 'HSV':
        image = image_normalize_hsv(image)
        
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
    x= np.array(x).astype('uint8')
    y = np.array(y).astype('uint8')
    
    full_data = np.zeros([x.shape[0], 4])
    full_data[:, :3] = x
    full_data[:, 3] = y
    
    full_data = pd.DataFrame(full_data)
    not_windshield = full_data[full_data[3] == 0]
    windshield = full_data[full_data[3] == 1]
    not_windshield = not_windshield.sample(n = int(1.5 * windshield.shape[0]))
    
    good_data = pd.concat([windshield, not_windshield], ignore_index = True)
    good_data = good_data.sample(frac = 1)
    
    X = good_data.values[:, 0:3]
    Y = good_data.values[:, 3]
    
    return X, Y

X = np.zeros([1, 3])
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

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 82)

from time import time
from sklearn.metrics import f1_score

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

from sklearn.neural_network import MLPClassifier
clf1 = MLPClassifier(hidden_layer_sizes = (50, 3), activation = 'logistic')

train_predict(clf1, X_train, Y_train, X_test, Y_test)

'''
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# lista de parâmetros a  calibrar
hidden_layers = []
for n_layers in range(1,5):
    for size in [10,20,30]:
        hidden_layers.append((size, n_layers))
parameters = {'hidden_layer_sizes':hidden_layers, 'activation':['logistic', 'tanh', 'relu']}

#função de pontuação f1 utilizando 'make_scorer' 
f1_scorer = make_scorer(f1_score)

clf = MLPClassifier( )
# Executa uma busca em matriz no classificador utilizando o f1_scorer como método de pontuação
grid_obj = GridSearchCV(clf, parameters)

#Ajusta o objeto de busca em matriz para o treinamento de dados e encontra os parâmetros ótimos
grid_obj.fit(X_train, Y_train)

# Get the best  estimator
clf = grid_obj.best_estimator_

parametros_otimos = clf.get_params()
parametros_otimos
#Results
# Imprime os resultados das estimativas de ambos treinamento e teste
print ("Pontuação F1 para o conjunto de treino: {:.4f}.".format(predict_labels(clf, X_train, Y_train)))
print ("Pontuação F1 para o conjunto de teste: {:.4f}.".format(predict_labels(clf, X_test, Y_test)))
'''
def plot_img(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#filename = '00002622327707790075353C1160220171731181805169ESREF161080080080097O000000011072016'

def aplicar_mascara(filename, color_map):
    dir_images = 'test_images' #diretorio que contem as imagens a serem analisadas
    image = cv2.imread(dir_images + '/' + filename + '.jpg', 1)
    image_predict = np.zeros([image.shape[0], image.shape[1]])
    #image_predict.astype('float')
    if color_map == 'HSV':
        image_norm = image_normalize_hsv(image)
    
    for i in range(30, 464):
        image_predict[i] = clf1.predict(image_norm[i])
    
    image_predict[:179, 396:] = 0
    plot_img(image_predict)
    
    kernel = cv2.imread('kernel.bmp', 0)
    kernel += 1
    image_predict = cv2.morphologyEx(image_predict, cv2.MORPH_OPEN, kernel)
    image_predict = cv2.morphologyEx(image_predict, cv2.MORPH_CLOSE, kernel)
    plot_img(image_predict)
    
    image_predict -= 1
    image_predict *= -1
    
    image_final = np.array(image)
    for i in range(3):
        image_final[:,:,i] = np.multiply(image_final[:,:,i], image_predict)
    dir_save = 'imagens_processadas'
    cv2.imwrite(os.getcwd() + '/' + dir_save + '/' + filename + '_proc.jpg', image_final)
    plot_img(image_final)

dir_images = 'small_test'#'test_images' #diretorio que contem as imagens a serem analisadas
files = [] # Ler os arquivos no diretorio
for (dirpath, dirnames, filenames) in os.walk(os.getcwd() + '/' + dir_images):#'/home/jesse/ProcImg/Projeto03'):
    files.extend(filenames)
    break

for file in files:
    file = file[:-4]
    aplicar_mascara(file, 'HSV')
    

def plot_img(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
plot_img(image)
    
#    plot_img(image_predict)