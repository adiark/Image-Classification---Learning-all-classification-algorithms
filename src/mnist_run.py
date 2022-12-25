# Importing tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Importing helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st

# Models
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm
import xgboost as xgb
import catboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

# Misc 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")

from src.resource import *


print(tf.__version__)

def run_boosting(selected_booster, X_train, y_train, X_test, y_test,class_names, selected_max_depth, selected_n_estimators,selected_way):
    
    prepare_running()

    def Train_boosting(clf, X_train, y_train, X_test, y_test):
        # Train
        model = clf.fit(X_train,y_train)
        # Predict
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, model.predict(X_test)))
        return (model,y_pred)

    def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
        y_pred = clf.predict(X_test)

        # structures
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:,i], y_pred == i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # roc for each class
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        sns.despine()
        return(fig)

    
    def plot_multiclass_pr(clf, X_test, y_test, n_classes, figsize=(17, 6)):
        y_pred = clf.predict(X_test)


        precision = dict()
        recall = dict()
            
        plt.show()

        y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
        
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_dummies[:, i],
                                                                y_pred[:] == i)
            plt.plot(recall[i], precision[i], lw=2, label='label {}'.format(i))
        
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall curve")
        
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        sns.despine()
        return(fig)


    def GridSearch_boosting(clf, params, X_train, y_train, X_test, y_test):
        model = GridSearchCV(clf, params, scoring='accuracy', n_jobs=-1, cv=5).fit(X_train,y_train).best_estimator_
        
        # Predict
        scores.append(accuracy_score(y_test, model.predict(X_test)))
        return model
    
    def tree_apply(clf):
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
            fig1 = plt.figure()
            sns.heatmap(cm, cmap="Blues", annot=True, yticklabels=class_names,  annot_kws={"fontsize":5},fmt='g', cbar=False)

            c_r = classification_report(y_test, y_pred, labels= clf.classes_,target_names=class_names, output_dict= True)
            fig2 = plt.figure()
            sns.heatmap(pd.DataFrame(c_r).iloc[:-1, :].T,cmap='Blues', annot=True)

            return (fig1, fig2)
        

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    def plot_image_f(i, predictions_array, true_label, img):
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        img1 = img[i:i+1,].reshape(28,28)
        plt.imshow(img1, cmap=plt.cm.binary, aspect="auto")

        predicted_label = predictions_array[i]
        if predicted_label == true_label[i]:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} class {} ({})".format(class_names[predicted_label],
                                             predictions_array[i],
                                             class_names[true_label[i]]), color=color)

    if selected_booster == "XGB":
        scores = []

        if selected_way == "Selected Parameters":
            xgboost,y_pred = Train_boosting(XGBClassifier(n_estimators=selected_n_estimators, max_depth=selected_max_depth), X_train, y_train, X_test, y_test)
        else :
            param_grid=[{'max_depth':[5,10],
            'n_estimators':[50],
            'learning_rate':[0.05,0.1],
            'colsample_bytree':[0.8,0.95]}]
            xgboost,y_pred = Train_boosting(XGBClassifier(n_estimators=100, max_depth=10), X_train, y_train, X_test, y_test)
        model = xgboost
        
        fig1, fig2 = tree_apply(xgboost)
        models = [('XGBoost', xgboost)]

        model_scores = pd.DataFrame({ 'Model': [name for name, _ in models], 'Test Accuracy': scores })
        model_scores.sort_values(by='Test Accuracy',ascending=False,inplace=True)
        
        fig3 = plot_multiclass_roc(xgboost, X_test, y_test, n_classes=10, figsize=(16, 10))
        fig4 = plot_multiclass_pr(xgboost, X_test, y_test, n_classes=10, figsize=(16, 10))

    elif selected_booster == "Decision Tree":
        scores = []

        if selected_way == "Selected Parameters":
            dt,y_pred = Train_boosting(DecisionTreeClassifier(max_features=selected_n_estimators, max_depth=selected_max_depth), X_train, y_train, X_test, y_test)
        else :
            # param_grid=[{'max_depth':[5,7,10],
            #     'max_features':[50]}]

            dt,y_pred = Train_boosting(DecisionTreeClassifier(max_features=100, max_depth=10), X_train, y_train, X_test, y_test)
        model = dt
        
        fig1, fig2 = tree_apply(dt)
        models = [('Decision Tree', dt)]

        model_scores = pd.DataFrame({ 'Model': [name for name, _ in models], 'Test Accuracy': scores })
        model_scores.sort_values(by='Test Accuracy',ascending=False,inplace=True)
        

        fig3 = plot_multiclass_roc(dt, X_test, y_test, n_classes=10, figsize=(16, 10))
        fig4 = plot_multiclass_pr(dt, X_test, y_test, n_classes=10, figsize=(16, 10))

    else :
        scores = []
        
        if selected_way == "Selected Parameters":
            lgb,y_pred = Train_boosting(LGBMClassifier(n_estimators=selected_n_estimators, max_depth=selected_max_depth), X_train, y_train, X_test, y_test)
        else:
            # param_grid=[{'max_depth':[5,10],
            #     'n_estimators':[50],
            #     'learning_rate':[0.05,0.1],
            #     'colsample_bytree':[0.8,0.95]}]
            lgb,y_pred = Train_boosting(LGBMClassifier(n_estimators=100, max_depth=10), X_train, y_train, X_test, y_test)
        model = lgb
        
        fig1, fig2 = tree_apply(lgb)
        models = [('LightGBM', lgb)]

        model_scores = pd.DataFrame({ 'Model': [name for name, _ in models], 'Test Accuracy': scores })
        model_scores.sort_values(by='Test Accuracy',ascending=False,inplace=True)
        
        fig3 = plot_multiclass_roc(lgb, X_test, y_test, n_classes=10, figsize=(16, 10))
        fig4 = plot_multiclass_pr(lgb, X_test, y_test, n_classes=10, figsize=(16, 10))

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    fig5 = plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))  
    for i in range(num_images):
        r = random.randint(0,len(X_test))
        if i == 0 :
            r1 = r
        plt.subplot(num_rows, num_cols, i + 1)
        plot_image_f(r, y_pred, y_test, X_test)
    fig5.tight_layout() 
    
    return (model_scores, fig1, fig2, fig3, fig4, fig5)


def run_classification_pca(selected_classifier, X_train, y_train, X_test, y_test,class_names):
    
    def plot_image_pca(i, predictions_array, true_label, img):
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        img1 = img[i:i+1,].reshape(28,28)
        plt.imshow(img1, cmap=plt.cm.binary, aspect="auto")

        predicted_label = predictions_array[i]
        if predicted_label == true_label[i]:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} class {} ({})".format(class_names[predicted_label],
                                             predictions_array[i],
                                             class_names[true_label[i]]), color=color)

    # applying pca
    prepare_running()
    pca = PCA(n_components=0.9,copy=True, whiten=False)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    # print(pca.explained_variance_ratio_)

    X_train_inv = pca.inverse_transform(X_train)
    X_test_inv = pca.inverse_transform(X_test)
    
    
    var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
    fig1 = go.Figure(data=go.Scatter(x=list(range(1,len(var)+1)), y=var))
    fig1.update_layout(xaxis_title='# Of Features',
                   yaxis_title='% Variance Explained')

    def classification_app(clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_pred =accuracy_score(y_test,y_pred)
        print("Accuracy on Test set:{:.4f}".format(acc_pred))

        con_matrix = pd.crosstab(pd.Series(y_test, name='Actual' ),pd.Series(y_pred, name='Predicted'))
        fig2 = plt.figure(figsize = (9,6))
        sns.heatmap(con_matrix, cmap="Blues",yticklabels=class_names, xticklabels=class_names, annot=True, fmt='g')
        return(fig2,acc_pred,y_pred,clf)

    if selected_classifier == "KNN":
        fig2, acc_pred, predictions, model = classification_app(KNeighborsClassifier()) 
        models = [('KNN')]
        model_scores = pd.DataFrame({ 'Model': models, 'Test Accuracy': acc_pred })
        
    elif selected_classifier == "SVC":
        fig2, acc_pred, predictions, model = classification_app(svm.SVC()) 
        models = [('SVM - Classifier')]
        model_scores = pd.DataFrame({ 'Model': models, 'Test Accuracy': acc_pred })
        
    elif selected_classifier == "Gaussian Naive Bayes":
        fig2, acc_pred, predictions, model = classification_app(GaussianNB()) 
        models = [('Gaussian Naive Bayer')]
        model_scores = pd.DataFrame({ 'Model': models, 'Test Accuracy': acc_pred })
        
    elif selected_classifier == "Logistic Regression":
        fig2, acc_pred, predictions, model = classification_app(LogisticRegression()) 
        models = [('Logistic Regression')]
        model_scores = pd.DataFrame({ 'Model': models, 'Test Accuracy': acc_pred })
        
    else :
        fig2, acc_pred, predictions, model = classification_app(RandomForestClassifier()) 
        models = [('Random Forest')]
        model_scores = pd.DataFrame({ 'Model': models, 'Test Accuracy': acc_pred })


    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    fig3 = plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))  
    for i in range(num_images):
        r = random.randint(0,len(X_test_inv))
        if i == 0 :
            r1 = r
        plt.subplot(num_rows,  num_cols, i + 1)
        plot_image_pca(r, predictions, y_test, X_test_inv)
    fig3.tight_layout() 
    
    return (model_scores,fig1, fig2,fig3)
    
def plot_image(i, predictions_array, true_label, img):
        X_train, y_train, X_test, y_test, class_names = load_data()
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img[i], cmap=plt.cm.binary, aspect="auto")

        predicted_label = np.argmax(predictions_array[i])
        if predicted_label == true_label[i]:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array[i]),
                                             class_names[true_label[i]]), color=color)

def plot_value_array(i, predictions_array, true_label):
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array[i], color="#777777")
        plt.ylim([0, 1])
        plt.xticks([])
        predicted_label = np.argmax(predictions_array[i])

        thisplot[predicted_label].set_color('red')
        thisplot[true_label[i]].set_color('blue')


def run_mnist(selected_optimizer, selected_metric, selected_epochs):
    prepare_running()
    X_train, y_train, X_test, y_test, class_names = load_data()
    col1, col2 = st.columns((1, 1))

    selections = f'{selected_optimizer} {selected_metric} {selected_epochs}'

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=selected_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=[selected_metric])

    model.fit(X_train, y_train, epochs=selected_epochs)  # epochs=5

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('\nTest Accuracy', test_acc)

    predictions = model.predict(X_test)

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    fig5 = plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))  
    for i in range(num_images):
        r = random.randint(0,len(X_test))
        if i == 0 :
            r1 = r
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(r, predictions, y_test, X_test)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(r, predictions, y_test)
    fig5.tight_layout() 
    
    img = X_test[r1]

    img = (np.expand_dims(img, 0))

    predictions_single = model.predict(img)

    fig6 = plt.figure(figsize=(8, 6)) # 5, 3
    plot_value_array(0, predictions_single, y_test[r1:r1+1]) 
    plt.title(f'Model Accuracy {"{:.2f}".format(test_acc)}')
    fig6.tight_layout() # For layout
    _ = plt.xticks(range(10), class_names, rotation=45)

    if selections not in SAVE_IMAGES:
        SAVE_IMAGES[selections] = [fig5, fig6]

    print(SAVE_IMAGES)
    st.success("Completed running model")

@st.cache
def load_data():
    mnist = keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return X_train, y_train, X_test, y_test, class_names

@st.cache
def load_datacsv():
    fetch_from = 'fashion-mnist_train.csv'
    train = pd.read_csv(fetch_from)

    fetch_from = 'fashion-mnist_test.csv'
    test = pd.read_csv(fetch_from)
    X_train, y_train, X_test, y_test = train.iloc[:,1:], train['label'], test.iloc[:,1:], test['label']

    # due to restricted resource allocation using subset of data
    # randomly generating subset 
    # random_index_X_train = X_train.index.values.tolist()
    # random_values_X_train = random.sample(random_index_X_train, 20000)
    # X_train = X_train.loc[random_values_X_train]
    # y_train = y_train.loc[random_values_X_train]

    # random_index_X_test = X_test.index.values.tolist()
    # random_values_X_test = random.sample(random_index_X_test, 3000)
   
    # X_test = X_test.loc[random_values_X_test]
    # y_test = y_test.loc[random_values_X_test]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return X_train, y_train, X_test, y_test, class_names



@st.cache
def prepare_running():
    # Below code is for "failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def show_data(X_train):
    st.subheader("A random image from the training set")
    fig1 = plt.figure(figsize=(8, 8))
    plt.imshow(X_train[random.randint(0,len(X_train))], aspect="auto")
    plt.colorbar()
    plt.grid(False)
    st.pyplot(fig1)

def show_data_labels(X_train, y_train, class_names):
    st.subheader("20 random images with class name")
    # 20 images with the class name
    fig2 = plt.figure(figsize=(8, 8))
    for i in range(20):
        plt.subplot(5, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        rand = random.randint(0,len(X_train))
        plt.imshow(X_train[rand], cmap=plt.cm.binary, aspect="auto")
        plt.xlabel(class_names[y_train[rand]])
    st.pyplot(fig2)


