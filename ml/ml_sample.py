import autosklearn.classification
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import shap
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

pass_score = 50
data_file = 'data/students_performance.csv'


def process():
    data = pd.read_csv(data_file)
    data = preprocess(data)
    data = encode_data(data)
    x_train, x_test, y_train, y_test = prepare_data(data)
    knn = train_with_knn(x_train, y_train, x_test, y_test)
    svm = train_with_svm(x_train, y_train, x_test, y_test)
    dtree = train_with_dtree(x_train, y_train, x_test, y_test)
    mlp = train_with_nn(x_train, y_train, x_test, y_test)
    gnb = train_with_naive_bayes(x_train, y_train, x_test, y_test)
    rf = train_with_random_forest(x_train, y_train, x_test, y_test)
    # automl = train_with_automl(x_train, y_train, x_test, y_test)

    models = dict({'KNN': knn, 'SVM': svm, 'Decision Tree': dtree, 'MLP': mlp, 'Naive Bayes': gnb, 'Random Forest': rf})

    vot = train_with_ensemble(models, x_train, y_train, x_test, y_test)
    models['Ensemble Voting'] = vot

    show_auc_curve_of_models(models, x_test, y_test)

    # explain_model(knn, x_test[:5])


def preprocess(data):
    data = data.drop(columns=['lunch'])
    data['result'] = data.apply(lambda x: 'fail' if (x['math score'] < pass_score or
                                                     x['reading score'] < pass_score or
                                                     x['writing score'] < pass_score) else 'pass', axis=1)
    return data


def encode_data(data):
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'])
    data['race/ethnicity'] = le.fit_transform(data['race/ethnicity'])
    data['parental level of education'] = le.fit_transform(data['parental level of education'])
    data['test preparation course'] = le.fit_transform(data['test preparation course'])
    data['result'] = le.fit_transform(data['result'])
    return data


def prepare_data(data):
    x = data.values[:, :7]
    y = data.values[:, 7]

    # normalize data
    mms = MinMaxScaler()
    x = mms.fit_transform(x)

    # split dataset into train, test (80%, 20%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

    return x_train, x_test, y_train, y_test


def train_with_knn(x_train, y_train, x_test, y_test):
    print("---Train with KNN---")

    # training with default hyper parameters
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    print("Accuracy:", knn.score(x_test, y_test))

    # use grid search to find the best set of hyper parameters
    params = dict(n_neighbors=np.arange(1, 10))
    best_estimator = grid_search(KNeighborsClassifier(), params, 5, x_train, y_train)

    # validation curve for n_neighbors hyper parameter
    param_range = np.arange(1, 10)
    mean_train_score, mean_test_score = calc_validation_curve(KNeighborsClassifier(), "n_neighbors", param_range,
                                                              5, x_train, y_train)

    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Validation Curve with KNN", "Number of Neighbours", "Accuracy")

    # learning curve for the best set of hyper parameters above
    sizes, mean_train_score, mean_test_score = calc_learning_curve(best_estimator, 5, x_train, y_train)

    plt.plot(sizes, mean_train_score, label="Training Score", color='b')
    plt.plot(sizes, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Learning Curve with KNN", "Training Set Size", "Accuracy")

    # confusion matrix for testing the best set of hyper parameters above
    show_confusion_matrix(best_estimator, x_train, y_train, x_test, y_test)

    return best_estimator


def train_with_svm(x_train, y_train, x_test, y_test):
    print("---Train with SVM---")

    # training with default hyper parameters
    svm = SVC()
    svm.fit(x_train, y_train)
    print("Accuracy:", svm.score(x_test, y_test))

    # use grid search to find the best set of hyper parameters
    params = dict(C=np.logspace(-2, 10, 13), gamma=np.logspace(-9, 3, 13))
    best_estimator = grid_search(SVC(probability=True), params, 5, x_train, y_train)

    # validation curve for C hyper parameter
    param_range = np.logspace(-2, 10, 13)
    mean_train_score, mean_test_score = calc_validation_curve(SVC(), "C", param_range,
                                                              5, x_train, y_train)

    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Validation Curve with SVM", "C", "Accuracy")

    # validation curve for gamma hyper parameter
    param_range = np.logspace(-9, 3, 13)
    mean_train_score, mean_test_score = calc_validation_curve(SVC(), "gamma", param_range,
                                                              5, x_train, y_train)

    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Validation Curve with SVM", "gamma", "Accuracy")

    # learning curve for the best set of hyper parameters above
    sizes, mean_train_score, mean_test_score = calc_learning_curve(best_estimator, 5, x_train, y_train)

    plt.plot(sizes, mean_train_score, label="Training Score", color='b')
    plt.plot(sizes, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Learning Curve with SVM", "Training Set Size", "Accuracy")

    # confusion matrix for testing the best set of hyper parameters above
    show_confusion_matrix(best_estimator, x_train, y_train, x_test, y_test)

    return best_estimator


def train_with_dtree(x_train, y_train, x_test, y_test):
    print("---Train with Decision Tree---")

    # training with default hyper parameters
    dtree = DecisionTreeClassifier()
    dtree.fit(x_train, y_train)
    print("Accuracy:", dtree.score(x_test, y_test))

    # use grid search to find the best set of hyper parameters
    params = dict(criterion=['gini', 'entropy'], max_depth=np.arange(1, 10))
    best_estimator = grid_search(DecisionTreeClassifier(), params, 5, x_train, y_train)

    # validation curve for max_depth hyper parameter
    param_range = np.arange(1, 10)
    mean_train_score, mean_test_score = calc_validation_curve(DecisionTreeClassifier(), "max_depth", param_range,
                                                              5, x_train, y_train)

    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Validation Curve with Decision Tree", "Max depth", "Accuracy")

    # learning curve for the best set of hyper parameters above
    sizes, mean_train_score, mean_test_score = calc_learning_curve(best_estimator, 5, x_train, y_train)

    plt.plot(sizes, mean_train_score, label="Training Score", color='b')
    plt.plot(sizes, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Learning Curve with Decision Tree", "Training Set Size", "Accuracy")

    # confusion matrix for testing the best set of hyper parameters above
    show_confusion_matrix(best_estimator, x_train, y_train, x_test, y_test)

    return best_estimator


def train_with_nn(x_train, y_train, x_test, y_test):
    print("---Train with Neural Network---")

    # training with default hyper parameters
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    print("Accuracy:", mlp.score(x_test, y_test))

    # use grid search to find the best set of hyper parameters
    params = dict(hidden_layer_sizes=[(1,), (2,), (3,)],
                  activation=['logistic', 'tanh', 'relu'])
    best_estimator = grid_search(MLPClassifier(), params, 5, x_train, y_train)

    # validation curve for hidden_layer_sizes hyper parameter
    param_range = [(1,), (2,), (3,)]
    mean_train_score, mean_test_score = calc_validation_curve(MLPClassifier(), "hidden_layer_sizes", param_range,
                                                              5, x_train, y_train)

    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Validation Curve with MLP", "Hidden layer sizes", "Accuracy")

    # learning curve for the best set of hyper parameters above
    sizes, mean_train_score, mean_test_score = calc_learning_curve(best_estimator, 5, x_train, y_train)

    plt.plot(sizes, mean_train_score, label="Training Score", color='b')
    plt.plot(sizes, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Learning Curve with MLP", "Training Set Size", "Accuracy")

    # confusion matrix for testing the best set of hyper parameters above
    show_confusion_matrix(best_estimator, x_train, y_train, x_test, y_test)

    return best_estimator


def train_with_naive_bayes(x_train, y_train, x_test, y_test):
    print("---Train with Naive Bayes---")

    # training with default hyper parameters
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print("Accuracy:", gnb.score(x_test, y_test))

    # use grid search to find the best set of hyper parameters
    params = dict(var_smoothing=np.logspace(-10, 1, 10))
    best_estimator = grid_search(GaussianNB(), params, 5, x_train, y_train)

    # validation curve for var_smoothing hyper parameter
    param_range = np.logspace(-10, 1, 10)
    mean_train_score, mean_test_score = calc_validation_curve(GaussianNB(), "var_smoothing", param_range,
                                                              5, x_train, y_train)

    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Validation Curve with Naive Bayes", "Var smoothing", "Accuracy")

    # learning curve for the best set of hyper parameters above
    sizes, mean_train_score, mean_test_score = calc_learning_curve(best_estimator, 5, x_train, y_train)

    plt.plot(sizes, mean_train_score, label="Training Score", color='b')
    plt.plot(sizes, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Learning Curve with Naive Bayes", "Training Set Size", "Accuracy")

    # confusion matrix for testing the best set of hyper parameters above
    show_confusion_matrix(best_estimator, x_train, y_train, x_test, y_test)

    return best_estimator


def train_with_random_forest(x_train, y_train, x_test, y_test):
    print("---Train with Random Forest---")

    # training with default hyper parameters
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    print("Accuracy:", rf.score(x_test, y_test))

    # use grid search to find the best set of hyper parameters
    params = dict(max_depth=np.arange(1, 10))
    best_estimator = grid_search(RandomForestClassifier(), params, 5, x_train, y_train)

    # validation curve for max_depth hyper parameter
    param_range = np.arange(1, 10)
    mean_train_score, mean_test_score = calc_validation_curve(RandomForestClassifier(), "max_depth", param_range,
                                                              5, x_train, y_train)

    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Validation Curve with Random Forest", "Max depth", "Accuracy")

    # learning curve for the best set of hyper parameters above
    sizes, mean_train_score, mean_test_score = calc_learning_curve(best_estimator, 5, x_train, y_train)

    plt.plot(sizes, mean_train_score, label="Training Score", color='b')
    plt.plot(sizes, mean_test_score, label="Cross Validation Score", color='g')
    show(plt, "Learning Curve with Random Forest", "Training Set Size", "Accuracy")

    # confusion matrix for testing the best set of hyper parameters above
    show_confusion_matrix(best_estimator, x_train, y_train, x_test, y_test)

    return best_estimator


def train_with_ensemble(models, x_train, y_train, x_test, y_test):
    print("---Train with Voting Classifier---")

    estimators = []
    for name in models:
        estimators.append([name, models[name]])

    vot = VotingClassifier(estimators=estimators, voting='soft')
    vot.fit(x_train, y_train)
    print("Accuracy:", vot.score(x_test, y_test))

    return vot


def train_with_automl(x_train, y_train, x_test, y_test):
    print("---Train with AutoML---")

    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(x_train, y_train)
    print("Accuracy:", automl.score(x_test, y_test))

    show_modes_str = automl.show_models()
    sprint_statistics_str = automl.sprint_statistics()

    print(show_modes_str)
    print(sprint_statistics_str)

    return automl


def show_auc_curve_of_models(models, x_test, y_test):
    for name in models:
        # predict probabilities & keep probabilities for the positive outcome only
        y_probs = models[name].predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probs)
        print("AUC for {} model: {}".format(name, roc_auc))
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        rgb = np.random.rand(3, )
        plt.plot(fpr, tpr, label=f"{name}, AUC={roc_auc:.2f}", color=rgb)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    show(plt, "AUC Curve", "False Positive Rate", "True Positive Rate")


def grid_search(model, params, cv, x_train, y_train):
    gs = GridSearchCV(model, param_grid=params, cv=cv, n_jobs=-1)
    gs_result = gs.fit(x_train, y_train)

    print("Best Params:", gs_result.best_params_)
    print("Best Score:", gs_result.best_score_)
    print("Best Estimator:", gs_result.best_estimator_)

    return gs_result.best_estimator_


def calc_validation_curve(model, param_name, param_range, cv, x_train, y_train):
    train_score, test_score = validation_curve(model, x_train, y_train,
                                               param_name=param_name,
                                               param_range=param_range,
                                               cv=cv, n_jobs=-1, scoring="accuracy")
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    return mean_train_score, mean_test_score


def calc_learning_curve(model, cv, x_train, y_train):
    sizes, train_score, test_score = learning_curve(model, x_train, y_train,
                                                    cv=cv, n_jobs=-1, scoring='accuracy',
                                                    train_sizes=np.linspace(0.01, 1.0, 50))
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    return sizes, mean_train_score, mean_test_score


def explain_model(model, x_test):
    explainer = shap.KernelExplainer(model.predict_proba, x_test)
    shap_values = explainer.shap_values(x_test)
    shap.plots.waterfall(shap_values[0])


def show_confusion_matrix(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Greens')
    plt.show()


def show(plot, title, x_label, y_label):
    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.grid()
    plot.tight_layout()
    plot.legend(loc='best')
    plot.show()


process()
