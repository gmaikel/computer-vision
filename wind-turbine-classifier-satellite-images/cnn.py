# Projet airbus de deep learning - CNN
# importation des packages

# Bibliothèques pour le modèle
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow as tf 

# Bibliothèques pour la matrice de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sn;sn.set(font_scale=1.4)

# Bibliothèques générales
import matplotlib.pyplot as plt 
import matplotlib.image as img 
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import cv2  

# %%
#verification_GPU

verification_GPU = tf.config.experimental.list_physical_devices('GPU')
print("GPU ok" if len(verification_GPU) == 1 else "Pas de GPU")

# %%
# Data loader

datasets = ['../input/airbus-data/train', '../input/airbus-data/val']

classes = ['background', 'target']
indice_classes = {'background': 0,  'target': 1}
nb_classes = 2

taille_image = (128,128)

output = []

# %%
"""
J'ai eu un problème avec la quantité de données :
(Connexion lente + message d'erreur : "Your notebook tried to allocate more memory than is available. It has restarted.").
Je n'ai pas réussi à résoudre le problème de la mémoire Ram d'une autre manière que de réduire la taille des données.

Solution : j'utilise seulement 15 000 images de test et 3 000 images de validation.

* En mode CPU ==> Ram = 16GB
* En mode GPU ==> Ram = 13GB
"""

def chargement_data():

    cpt = 0
    for dataset in datasets:
        print(dataset)
        images = []
        labels = []
                
        for folder in classes:
            print(folder)
            label = indice_classes[folder]
            
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                #print(file)
                #print(os.listdir(os.path.join(dataset, folder)))
                
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, taille_image) 
                
                images.append(image)
                labels.append(label)
                
                cpt += 1
                if dataset == '../input/airbus-data/val':
                    n = 1500
                else:
                    n = 7500
                if cpt == n:
                    break
                    
            cpt = 0
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))
        
    print("Fin du chargement")
    return output

# %%
# chargements des données : Données d'apprentissage & Données de validation (ratio = 20%)

(train_images, train_indices), (val_images, val_indices) = chargement_data()

# %%
# Data analysis / Data visualization

def affichage_image_random(fichier, fichier_indices):
    """
        Permet d'afficher une image au hasard d'un dataset.
    """
        
    i = np.random.randint(fichier.shape[0])
    plt.figure()
    plt.imshow(fichier[i])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image {}# : '.format(i) + classes[fichier_indices[i]])
    plt.show()
    
# %%
# Exemple
affichage_image_random(train_images, train_indices)

# %%
# Quantité 
nbr_train = train_indices.shape[0]
nbr_test = val_indices.shape[0]
print ("Quantité de données d'apprentissage: ", nbr_train)
print ("Quantité de données de validation: ", nbr_test)
print ("Taille d'une image: ", taille_image,'\n')


# Affichage en diagramme 
m,train_q = np.unique(train_indices, return_counts=True)
m,val_q = np.unique(val_indices, return_counts=True)
pd.DataFrame({'train': train_q,'val': val_q}, index=classes).plot.bar()
plt.show()

# %%
# Normaliser les données poour mieux les comparer

print("Max avant normalisation : ",train_images.max())

train_images = train_images / 255.0 
val_images = val_images / 255.0

print("Max après normalisation : ",train_images.max())

# %%
# Création du modèle CNN

model = Sequential()

# 1ere couche (convolution & polling)
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (128, 128, 3)))
model.add(MaxPooling2D(pool_size = (2,2))) 

# 2eme couche (convolution & polling)
model.add(Conv2D(64, (3,3), activation ='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

## 3eme couche (convolution & polling)
model.add(Conv2D(64, (3,3), activation ='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

# 4eme couche (convolution & polling)
model.add(Conv2D(64, (3,3), activation ='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


# Construction d'un vecteur à partir de la matrice
model.add(Flatten())

# Deux couches dense
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation = "softmax")) # "softmax" pour l'activation, pour avoir la probabilité entre nos deux classes

# %%
model.summary()
 
# %%
# Compilation 

model.compile(optimizer = Adam(learning_rate=0.0005), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# %%
# Entraintement du modèle

history = model.fit(train_images, train_indices, batch_size=32, epochs=10, validation_split=0.2)

# %%
# Evaluation du modèle 
"""
    Renvoie la valeur de perte et la valeur passée en "metrics", ici : Accuracy (donc la précision du modèle)
"""

test_modele = model.evaluate(val_images, val_indices)
print("\nLa précision du modèle : ", round(test_modele[1], 2), "\nlearning_rate : 0.0005", "\nbatch_size : 32", "\nepochs : 10")

# Une autre méthode pour avoir la précision du modèle
#from sklearn.metrics import accuracy_score
#print("Accuracy : {}".format(accuracy_score(val_indices, pred_indices)))

# %%
# Visualisaiton des résultats 

def plot_accuracy_loss(history):
    """
        Plot accuracy/loss, objectif :  constater l'overfitting / underfitting.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("train accuracy vs val_accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss 
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()
    print('\n')
    

# %%
# Affichage courbe 

plot_accuracy_loss(history)

#history.history?? #Afficher les infos après entrainement

# %%
# Test 

"""
    On prédit les images dans le dataset val_images.
    On obtient un vecteur de probabilité, on prend la probabilité la plus grande.
"""
    
predictions = model.predict(val_images)    
pred_indices = np.argmax(predictions, axis = 1) 

# %%
# Matrice de confusion
#8 epoch, 0.001, 32
CM = confusion_matrix(val_indices, pred_indices)
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 15}, 
           xticklabels=classes, 
           yticklabels=classes, ax = ax)
ax.set_title('Matrice de confusion')
plt.show()

# %%
# Prédire une image choisit au hasard 

affichage_image_random(val_images, pred_indices)

# %%
# Recherche des images mal classifiées

def exemples_images(dataset, dataset_indices):
    """
       Cette focntion permet d'afficher 25 images, sera utilisée pour afficher les images mal classifiées.
    """
    
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Voici un exemple de 25 images", fontsize=15)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(dataset[i], cmap=plt.cm.binary)
        plt.xlabel(classes[dataset_indices[i]])
    plt.show()
    

def images_mal_classifiees(test_images, test_labels, pred_labels):
    """
        Cette fonction va nous renvoyer un échantillon de 25 images mal classifiées ==> test_labels != pred_labels
    """
    BOO = (test_labels == pred_labels)
    index = np.where(BOO == 0)
    malclass_images = test_images[index]
    malclass_indice = pred_labels[index]

    title = "Exemples d'images mal classifiées:"
    exemples_images(malclass_images, malclass_indice)
    
# %%
# Affichage de quelques images mal classifiées

images_mal_classifiees(val_images, val_indices, pred_indices)
