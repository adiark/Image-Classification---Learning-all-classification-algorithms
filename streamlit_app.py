import streamlit as st
import urllib
from src.mnist_run import *
from src.resource import * 

def main():
    st.set_page_config(page_title="Fashion MNIST with Streamlit", layout="wide")

    sidebar_menu_ui()

def sidebar_menu_ui():
    st.sidebar.title(SIDEBAR_TITLE)
    app_mode = st.sidebar.selectbox("Please choose an option",
        [SIDEBAR_OPTION_0,SIDEBAR_OPTION_1, SIDEBAR_OPTION_2, SIDEBAR_OPTION_3])
    if app_mode == SIDEBAR_OPTION_0:
        st.title(MAIN_TITLE)
        st.write("Welcome to the MNIST Fashion classification app! This app classify images of\
            different fashion items using multiple algorithms from scikit-learn and TensorFlow. The first section of the application describes the data\
            and visualize sample images with their classification labels. The second section of the app includes PCA and classification\
            algorithms such as LightGBM, XGBoost, Decision Tree, KNN, SVC, Random forest etc. with some functionality for user to select hyperparameters \
            and understand how the algorithm works. In the PCA-Classifier select option, the objective is to understand the power of Principal Component Analysis\
            (PCA) done on the data before performing the classification. We classify using defaut parameters and visualise the improvement in accuracy of our model.\
            In the third section of the application one can explore classification through tensorflow models, the goal is to create multiple models by selecting\
            hyperparameters and analyse the accuracy and prediction.")

        st.write("Once you choosing your desired algorithm and parameters, the app will train the model on the MNIST Fashion dataset\
            and evaluate its performance on test data using metrics such as the confusion matrix, classification report, ROC curve,\
            and precision-recall curve. You can then use the app to classify the images and see how well the model performs.\
            Try it out and see how accurate the different algorithms are at classifying fashion items!")
        
        with st.expander("Important Note"):
            st.write("Some algorithms are resource intensive like LightGBM, XGBoost and other ensemble based models,\
                this might occasionaly create application to crash due to overload resource allocation restriction on streamlit\
                Try refreshing the application in such instances.")

        st.write("Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 example\
                and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. First column \
                represent the label category and the rest contains pixel values from range(0-255).")
        show_raw_data()
        st.write(" Normalization in machine learning is the process of translating data into the range [0, 1] (or any other range) or simply transforming data onto the unit sphere. \
                Some machine learning algorithms benefit from normalization and standardization, particularly when Euclidean distance is used. For example, if one of the variables \
                in the K-Nearest Neighbor, KNN, is in the 1000s and the other is in the 0.1s, the first variable will dominate the distance rather strongly.\
                In this scenario, normalization and standardization might be beneficial.")
        show_normalised_data()
        show_brief_data()
    elif app_mode == SIDEBAR_OPTION_1:
        run_sk_app()
    elif app_mode == SIDEBAR_OPTION_2:
        run_app()
    elif app_mode == SIDEBAR_OPTION_3:
        st.code(get_file_content_as_string(APP_FILE_NAME))

def show_raw_data():
    col1,  = st.columns(1)
    with col1:
        with st.expander("Show raw data."):
            fetch_from = 'fashion-mnist_train.csv'
            train = pd.read_csv(fetch_from)
            st.dataframe(train.head(15))       

def show_normalised_data():
    col1,  = st.columns(1)
    
    with col1:
        with st.expander("Show normalised data."):
            X_train_csv, y_train_csv, X_test_csv, y_test_csv, class_names_csv = load_datacsv()
            X_train_csv = X_train_csv / 255.0
            st.dataframe(X_train_csv.head(15))
          

def show_brief_data():
    train_images, train_labels, test_images, test_labels, class_names = load_data()
    col1, col2 = st.columns((1, 1))

    with col1:
        with st.expander("Show a random data sample."):
            show_data(train_images)

    with col2:
        with st.expander("Show 20 images with labels."):
            show_data_labels(train_images, train_labels, class_names)

def run_sk_app():
    X_train_csv, y_train_csv, X_test_csv, y_test_csv, class_names_csv = load_datacsv()
    X_train_csv, y_train_csv, X_test_csv, y_test_csv = X_train_csv.to_numpy(), y_train_csv.to_numpy(), X_test_csv.to_numpy(), y_test_csv.to_numpy()
    st.title("Fashion MNIST data classification using SkLearn")
    SELECTED_BATTLE = battle_selector_ui()
    if SELECTED_BATTLE == "Boosting":
        st.subheader("Description Expanders")
        with st.expander("LightGBM & XGBoost"):
            st.write("LightGBM and XGBoost are both gradient boosting algorithms,\
            which means that they train decision trees in a sequential manner,\
            using the output of one tree to improve the next tree. This can \
            be thought of as a group of trees working together to make predictions,\
            with each tree learning from the mistakes of the previous trees. \
            The main advantage of gradient boosting algorithms is that they can \
            often produce very accurate predictions, and they are also quite fast to train.")
            st.image("XLGB.png")
        with st.expander("Decision Tree"):
            st.write("Decision trees, on the other hand, work by splitting the data into smaller \
            and smaller groups based on the values of the features in the data. \
            For example, if we are trying to predict whether a person will have a heart attack,\
            we might use features like age, gender, and cholesterol levels to split the data into groups.\
            The advantage of decision trees is that they are easy to understand and interpret, and they can\
            handle both numerical and categorical data. However, they can sometimes be less accurate than\
            other algorithms, and they can also be prone to overfitting if they are not used carefully.")
            st.image("DT.png")
            st.write("Overall, the choice of which algorithm to use will depend on the specific problem you are\
            trying to solve, and you will need to experiment with different algorithms to\
                see which one works best for your data.")

        with st.expander("How to select hyperparameters?"):
            st.write("The selection of hyperparameters, such as the maximum depth of the tree and the number of estimators,\
                can have a significant impact on the performance of a model. In general, the best way to select \
                these hyperparameters is through a process called grid search, which involves training the model with different\
                combinations of hyperparameters and evaluating each one using a metric such as accuracy or F1 score.\
                The hyperparameters that result in the best performance on the evaluation metric can then be chosen for the final model.")

            st.write("The maximum depth of a tree refers to the maximum number of levels in the tree.\
                For example, a tree with a maximum depth of 3 will have a maximum of 4 levels (including the root node),\
                with the first level consisting of the root node, the second level consisting of the child nodes of the root node,\
                and so on. The number of estimators, on the other hand, refers to the number of trees that are used in the model.\
                A model with a large number of estimators will have more trees, \
                which can improve its performance but can also make it more computationally expensive to train and use.")

            st.write("In decision trees, the maximum number of features(estimators) hyperparameter determines the maximum number of\
                features that the algorithm will consider when splitting the data into smaller groups. This hyperparameter\
                is used to control the complexity of the model, and setting it to a higher value can often improve the accuracy\
                of the model. However, using too many features can also lead to overfitting, so it's important to find the right balance.")
            
        with st.expander("How to measure accuracy?"):
            st.write("There are several ways to measure the accuracy of a classification model. One common metric is the \
                accuracy score, which is the fraction of correct predictions made by the model. Other metrics that are \
                commonly used to evaluate classification models include precision, recall, and the F1 score.\
                These metrics can be calculated using the true and predicted labels for the data that the model \
                was evaluated on. In general, you want to use a combination of different metrics to get a complete \
                picture of the performance of your model.")
            
        with st.expander("Visualization tools"):
            st.write("Confusion matrix and classification report are tools for evaluating the performance of a classification model.\
                A confusion matrix is a table that shows the number of correct and incorrect predictions made by the\
                model for each class. It is often used to visualize the model's performance and to identify the classes\
                where the model is making the most errors. A classification report is a text-based summary of the performance\
                of a classification model, which includes the precision, recall, and F1 score for each class.\
                Both the confusion matrix and the classification report can be useful for understanding how well\
                a model is performing and for identifying areas where it may be failing")
            
        with st.expander("ROC, Precision and Recall"):
            st.write("Receiver operating characteristic (ROC) curve and precision-recall curve are two commonly\
                used plots for evaluating the performance of binary classification models. An ROC curve plots the\
                true positive rate (TPR) against the false positive rate (FPR) at different threshold settings. \
                The TPR is the fraction of positive examples that are correctly classified, while the FPR is the \
                fraction of negative examples that are incorrectly classified. An ROC curve allows you to visualize\
                the trade-off between the TPR and FPR of a model, and it can be used to compare the performance of different models.")

            st.write("A precision-recall curve plots the precision and recall of a model at different threshold settings.\
                Precision is the fraction of correct positive predictions made by the model, while recall is the fraction of\
                positive examples that were correctly classified. A precision-recall curve allows you to visualize the trade-off\
                between precision and recall, and it can be useful for evaluating the performance of a model when the goal is to make\
                as many correct positive predictions as possible.")

        SELECTED_BOOSTER = boosting_selector_ui()
        SELECTED_DEPTH = max_depth_selector_ui()
        SELECTED_N_ESTIMATOR = n_estimator_ui()
        SELECTED_WAY = pg_selector_ui()

        if st.sidebar.button("Start Boosting"):   
            score, cm, cr, roc, pr, pd = run_boosting(SELECTED_BOOSTER,X_train_csv, y_train_csv, X_test_csv, y_test_csv, class_names_csv,SELECTED_DEPTH,SELECTED_N_ESTIMATOR,SELECTED_WAY)
            st.dataframe(score)
            col1, col2 = st.columns((1, 1))
            with col1:
                st.subheader("Confusion Matrix")
                st.pyplot(cm)
            with col2:
                st.subheader("Classification Report")
                st.pyplot(cr)
            
            col3, col4 = st.columns((1,1))
            with col3:
                st.subheader("ROC Curve")
                st.pyplot(roc)
            with col4:
                st.subheader("Precision Recall Curve")
                st.pyplot(pr) 
            
            st.subheader("Prediction Sample")
            st.pyplot(pd)   
    else:
        SELECTED_CLASSIFIER = classifier_selector_ui()
        st.subheader("Description Expanders")
        with st.expander("What is PCA?"):
            st.write("Principal component analysis (PCA) is a statistical technique that is used to reduce the\
                dimensionality of a dataset. It does this by transforming the data into a new set of variables,\
                called principal components, which are uncorrelated and ordered in such a way that the first principal\
                component has the highest possible variance, the second principal component has the second highest possible\
                variance, and so on. This transformation allows you to represent the same data using fewer dimensions, \
                which can be useful for visualizing the data or for building machine learning models that work with lower-dimensional data.")
            st.image("PCA.png")
            
        with st.expander("SVC"):
            st.write("One common variant of the SVM algorithm is the support vector classifier (SVC), \
                which is a binary classifier that uses the same principles as an SVM but only considers \
                two classes at a time. When working with multi-class data, an SVC model can be trained using\
                a one-versus-rest approach, in which a separate binary classifier is trained for each class and\
                the class that gets the highest confidence score is predicted.")
            st.image("SVC.png")
            
        with st.expander("KNeighbours"):
            st.write("K-nearest neighbors (KNN) is a type of supervised learning algorithm that can be used for classification or regression.\
                In the case of classification, the KNN algorithm works by using the training data to find the K examples \
                in the training set that are closest to the new example, and then using the labels of those examples to predict\
                the label of the new example. The number of neighbors, K, is a hyperparameter that you need to specify when training the model.")

            st.write("It is called a lazy learning algorithm because it does not build a model explicitly, \
                but instead makes predictions based on the training data. This means that the training time \
                for KNN is very fast, but the prediction time can be slow if the training set is large. \
                Additionally, the performance of KNN can be sensitive to the choice of K, and it can be affected \
                by the presence of noisy or irrelevant features in the data")
            st.image("KNN.png")

        with st.expander("Gaussian Naive Bayes"):
            st.write("The Gaussian naive Bayes algorithm is a classification algorithm that uses Bayes' theorem \
                to make predictions about the classes of a given set of data. Bayes' theorem states that the probability \
                of an event, given some evidence, is equal to the prior probability of the event multiplied by the likelihood\
                of the evidence, divided by the marginal probability of the evidence.")

            st.write("In the case of the Gaussian naive Bayes algorithm, the event in question is the class of a given example,\
                and the evidence is the values of the features of that example. The algorithm assumes that the features are all \
                independent of each other and that each feature follows a normal (Gaussian) distribution. Given these assumptions,\
                the algorithm can use Bayes' theorem to calculate the probabilities of each class for a given example, and then predict\
                the class with the highest probability as the final prediction.")

            st.write("The Gaussian naive Bayes algorithm is often used in applications where the data is \
                limited and the number of features is relatively small. It is a simple and fast algorithm \
                that can give good performance if the assumptions it makes about the data are valid. \
                However, it can be less effective if the data is complex and the assumptions it makes about the data are not satisfied.")
            st.image("GNB.png")
            
        with st.expander("Logistic Regression"):
            st.write("Logistic regression is a type of supervised learning algorithm that is used for classification.\
                It is a regression algorithm, but instead of predicting a continuous value, it predicts the probability\
                that an example belongs to a particular class. The predicted probability is calculated using a logistic\
                function, which maps the input values to a value between 0 and 1.")

            st.write("In logistic regression, a model is trained using a training set of labeled examples. \
                The model learns the relationship between the features of the examples and their labels, \
                and then uses that relationship to make predictions on new examples. The predicted probabilities \
                can be converted into class labels using a threshold value. For example, if the threshold is 0.5, \
                then examples with a predicted probability greater than 0.5 are predicted to belong to the positive class, \
                and examples with a predicted probability less than 0.5 are predicted to belong to the negative class.")

            st.write("Logistic regression is a simple and effective algorithm that can be used for binary and multi-class\
                classification. It is easy to implement and can give good performance if the data is linearly separable.\
                However, it can be less effective if the data is complex or non-linear.")
            st.image("LR.png")
            
        with st.expander("Random Forest"):
            st.write("Random forest is an ensemble learning algorithm that is used for classification and regression.\
                It is called a forest because it is made up of many decision trees, which are trained using the training\
                data and then combined to make predictions.")

            st.write("Each decision tree in a random forest is trained on a random subset of the training data, and uses a\
                random subset of the features of the data to make predictions. The final predictions made by the random \
                forest are the average of the predictions made by the individual decision trees. This combination of many\
                decision trees with random subsets of the data and features is what gives random forests their strength.\
                It allows the algorithm to capture the wisdom of the crowd and make more accurate predictions than any\
                individual decision tree.")

            st.write("Random forest is a powerful and flexible algorithm that can give good performance on a wide range\
                of tasks. It is relatively easy to train and can handle data with missing values and with a mix of numerical\
                and categorical features. However, it can be sensitive to the parameters that control the number of decision\
                trees and the subsets of data and features used to train each tree. Choosing these parameters can be challenging,\
                and it is often necessary to use a combination of trial and error and cross-validation to find the best values")
            st.image("RF.jpeg")

        if st.sidebar.button("Classify"):
            score, pca, cm, pred1 = run_classification_pca(SELECTED_CLASSIFIER,X_train_csv, y_train_csv, X_test_csv, y_test_csv, class_names_csv)
            st.dataframe(score)
            col1, col2 = st.columns((1, 1))
            with col1:
                st.subheader("PCA - Variance Explained")
                st.plotly_chart(pca)
            with col2:
                st.subheader("Confusion Matrix")
                st.pyplot(cm)
            
            st.subheader("Prediction Sample")
            st.pyplot(pred1)

    
def run_app():
    st.title("Fashion MNIST data classification using TensorFlow")
    SELECTED_OPTIMIZER = optimizer_selector_ui()
    SELECTED_METRIC = metric_selector_ui()
    SELECTED_EPOCH = epoch_selector_ui()
    if st.sidebar.button("Start classification"):
        run_mnist(selected_optimizer=SELECTED_OPTIMIZER, selected_metric=SELECTED_METRIC,
                  selected_epochs=SELECTED_EPOCH)
        selections = f'{SELECTED_OPTIMIZER} {SELECTED_METRIC} {SELECTED_EPOCH}'
        if selections not in HYPER_PARAMS:
            HYPER_PARAMS.append(selections)

    if HYPER_PARAMS: 
        selections = st.sidebar.multiselect("Choose the results and compare (Optimizer Metric Epoch)", HYPER_PARAMS)
        if selections:
            st.header("Comparison")
            compare_plots(selections)


@st.cache(show_spinner=False) 
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/adiark/Fashion-Image-Classification---MNIST/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def battle_selector_ui():
    battle_name = st.sidebar.selectbox("Select a Type",SELECTED_BATTLE, 0) # Default selection is Boosting
    return battle_name

def max_depth_selector_ui():
    max_depth = st.sidebar.slider("Please choose Maximum depth", 1, 15, MAXDEPTH) # Default selection is 5
    return max_depth
    
def n_estimator_ui():
    n_estimator = st.sidebar.slider("Please choose # of estimator", 1, 150, N_ESTIMATOR) # Default selection is 10
    return n_estimator

def pg_selector_ui():
    pg_name = st.sidebar.selectbox("Select a Type",SELECTED_WAY, 0) # Default selection is user
    return pg_name

def classifier_selector_ui():
    classifier_name = st.sidebar.selectbox("Please choose a Classifier", CLASSIFIERS, 0) # Default selection is SVM
    return classifier_name

def boosting_selector_ui():
    boosting_name = st.sidebar.selectbox("Please choose an Algorithm", BOOSTING, 0) # Default selection is LGB
    return boosting_name

def optimizer_selector_ui():
    st.sidebar.markdown("# Optimizer")
    optimizer_name = st.sidebar.selectbox("Please choose an optimizer", OPTIMIZERS, 0) # Default selection is Adam
    return optimizer_name

def metric_selector_ui():
    st.sidebar.markdown("# Metric")
    metric_name = st.sidebar.radio("Please choose a metric", list(METRICS.keys()), 0) # Default selection is Accuracy
    return METRICS[metric_name]

def epoch_selector_ui():
    st.sidebar.markdown("# Epoch")
    epochs_selection = st.sidebar.slider("Please choose the epoch", EPOCH_MIN, EPOCH_MAX, EPOCH_MIN)
    return int(epochs_selection)

def compare_plots(hyper_parameters):
    for key in hyper_parameters:
        if key in SAVE_IMAGES:
            parameters = key.split(' ')
            description = f'Optimizer: **{parameters[0]}**, Metric: **{parameters[1]}**, Epoch: **{parameters[2]}**'
            st.write(description)
            col1, col2 = st.columns((1, 1))
            with st.spinner("Loading..."):
                with col1:
                    st.pyplot(SAVE_IMAGES[key][0])

                with col2:
                    st.pyplot(SAVE_IMAGES[key][1])

if __name__ == "__main__":
    main()