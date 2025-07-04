# import necessary packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, RocCurveDisplay
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.preprocessing import LabelBinarizer 
from itertools import cycle

# ------------------------------------
# Page Setup
# ------------------------------------
st.set_page_config(
    page_title="Decision Tree Hyperparameter Tuning",
    page_icon=":deciduous_tree:",
    layout="wide"
)

# ------------------------------------
# Application Information
# ------------------------------------
st.title("Decision Tree Hyperparameter Tuning")
st.markdown("""
### About this Application
This interactive application demonstrates the performance of a Decision Tree classifier with different hyperparameters. You can:
- Input your own dataset or choose one of the demo datasets available.
- Choose a metric to use as the criterion for splitting the data at each node.
- Tune the model's hyperparameters.
- Automatically find the best tree according to a scoring parameter of your choice.
""")

# ------------------------------------
# Helper Functions
# ------------------------------------

# turn sklearn toy datasets into pandas dataframes
def toy_to_df(load_function):
    bunch = load_function()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df

# data preprocessing
def preprocess(df):
    # encode categorical variables
    categorical_cols = [col for col in df.columns
                        if (pd.api.types.is_categorical_dtype(df[col].dtype)
                        or pd.api.types.is_object_dtype(df[col].dtype))
                        and col != target]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # define features and target
    X = df.drop(target, axis=1)
    y = df[target]
    return df, X, y

# split dataset
def split_data(X, y, test_size = 0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# train decision tree
def train_decision_tree(X_train, y_train, criterion, max_depth, min_samples_split, min_samples_leaf):
    model = DecisionTreeClassifier(criterion=criterion,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=42)
    model.fit(X_train, y_train)
    return model

# use grid search cross-validation to find best decision tree
def find_best_tree(X_train, y_train, X_test, scoring):
    # define parameter grid
    param_grid = {
        "criterion" : ["gini", "entropy", "log_loss"],
        "max_depth" : range(1, 11),
        "min_samples_split" : range(2, 11, 2),
        "min_samples_leaf" : range(1, 11)
    }
    # initialize decision tree classifier
    dtree = DecisionTreeClassifier(random_state=42)
    # set up grid search cv
    grid_search = GridSearchCV(estimator=dtree,
                               param_grid=param_grid,
                               cv=5,
                               scoring=scoring)
    # fit grid search cv to the training data
    grid_search.fit(X_train, y_train)
    # get best parameters
    best_params = grid_search.best_params_
    # get best estimator
    best_dtree = grid_search.best_estimator_
    # predict on the test set
    y_pred = best_dtree.predict(X_test)
    return y_pred, best_params, best_dtree

# plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
    plt.clf()

# plot ROC and AUC
# thank you to Gemini for the help with this function
def plot_multiclass_roc(model, y_train, X_test, y_test, target_classes, target_names):
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_score = model.predict_proba(X_test)
    n_classes = len(target_classes)

    fig, ax = plt.subplots(figsize=(6, 6))

    if n_classes == 2:
        # binary classification case: plot a single ROC curve
        # assuming y_score has shape (n_samples, 1) and contains the probability of the positive class
        fpr, tpr, _ = roc_curve(y_onehot_test[:, 0], y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
    else:
        # multi-class classification case: plot macro-average and individual ROC curves
        # multi-class classification case: plot macro-average and individual ROC curves
        fpr = []
        tpr = []
        roc_auc = []

        for i in range(n_classes):
            f, t, _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
            fpr.append(f)
            tpr.append(t)
            roc_auc.append(auc(fpr[-1], tpr[-1]))

        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(n_classes):
            if i < len(fpr) and i < len(tpr) and len(fpr[i]) > 1 and len(tpr[i]) > 1:
                mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr_macro = fpr_grid
        tpr_macro = mean_tpr
        roc_auc_macro = auc(fpr_macro, tpr_macro)

        ax.plot(
            fpr_macro,
            tpr_macro,
            label=f"macro-average ROC curve (AUC = {roc_auc_macro:.2f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for class_id, color in zip(range(n_classes), colors):
            RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                y_score[:, class_id],
                name=f"ROC curve for {target_names[class_id]}",
                color=color,
                ax=ax,
                plot_chance_level=(class_id == n_classes - 1)
            )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
        )
        ax.legend(loc="lower right")

    st.pyplot(fig)

# plot the decision tree
def plot_decision_tree(model,df,target,X_train):
    dot_data = tree.export_graphviz(model,
                                    feature_names=X_train.columns,
                                    class_names = df[target].unique().astype('str'),
                                    filled=True)
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph)

# ------------------------------------
# Sidebar Structure
# ------------------------------------

# dataset upload option
st.sidebar.markdown("## Dataset Selection")
dataset_upload = st.sidebar.file_uploader(label="Upload your own dataset",
                                          type="csv")

# use uploaded dataset if user inputs one
if dataset_upload is not None:
    dataset = pd.read_csv(dataset_upload)
    target = st.sidebar.selectbox(label="What is the target variable?",
                                  options=dataset.columns,
                                  index=None)

# dataset demo option
dataset_demo = None
if dataset_upload is None:
    st.sidebar.markdown("#### No dataset? Use a demo")
    dataset_demo = st.sidebar.selectbox(label="Demo datasets",
                                        options=["Breast Cancer", "Iris", "Wine"],
                                        index=None)

# use demo datasets if user chooses one
if dataset_demo == "Breast Cancer":
    dataset = toy_to_df(load_breast_cancer)
    target = "target"
elif dataset_demo == "Iris":
    dataset = toy_to_df(load_iris)
    target = "target"
elif dataset_demo == "Wine":
    dataset = toy_to_df(load_wine)
    target = "target"

# select hyperparameters
st.sidebar.markdown("## Hyperparameter Selection")

# select hyperparameter: criterion
criterion = st.sidebar.selectbox(label="Criterion",
                                options=["Gini Index", "Entropy", "Log Loss"],
                                index=None)
if criterion == "Gini Index":
    criterion = "gini"
elif criterion == "Entropy":
    criterion = "entropy"
elif criterion == "Log Loss":
    criterion = "log_loss"

# select hyperparameter: max depth
max_depth = st.sidebar.select_slider(label="Maximum depth of tree",
                                    options=range(1, 11),
                                    value=1)

# select hyperparameter: min samples split
min_samples_split = st.sidebar.select_slider(label="Minimum samples to split",
                                            options=range(2,11,2),
                                            value=2)

# select hyperparemeter: min samples leaf
min_samples_leaf = st.sidebar.select_slider(label="Minimum samples for leaf",
                                            options=range(1, 11),
                                            value=1)

# show best decision tree
st.sidebar.markdown("## Best Decision Tree")

# select scoring metric for grid search
scoring_choice = st.sidebar.selectbox(label="Choose scoring metric",
                                     options=["Accuracy", "F1 Score", "Precision", "Recall"],
                                              index=None)

if scoring_choice == "Accuracy":
    scoring_metric = "accuracy"
elif scoring_choice == "F1 Score":
    scoring_metric = "f1"
elif scoring_choice == "Precision":
    scoring_metric = "precision"
elif scoring_choice == "Recall":
    scoring_metric = "recall"

# select to look for best tree with grid search
tree_finder = st.sidebar.toggle("Find the best decision tree for me")
st.sidebar.markdown("**Note:** This may take a few seconds to run.")
st.sidebar.caption("Changing the _Hyperparameter Selection_ buttons won't change the best decision tree.")

# ------------------------------------
# Main Page Structure
# ------------------------------------

# execute if user has selected a dataset and target variable
if (dataset_upload or dataset_demo is not None) and target is not None and criterion is not None:

    # preprocess and split df, train model
    processed_df, X, y = preprocess(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # show user-selected decision tree if they don't choose to see the best one
    if tree_finder is False:

        # fit model
        dt_model = train_decision_tree(X_train=X_train, y_train=y_train, criterion=criterion, max_depth=max_depth,
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

        # classification report
        st.subheader("Classification Report")

        # predict and evaluate model
        y_pred = dt_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {accuracy:.2f}")

        # display classification report
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df)

        # create two columns for side-by-side display
        col1, col2 = st.columns(2)

        # show confusion matrix on column 1 
        with col1:
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm)
        
        # show ROC and AUC plot on column 2
        with col2:
            st.subheader("ROC and AUC plot")
            plot_multiclass_roc(model=dt_model, y_train= y_train,X_test=X_test, y_test=y_test,
                                target_classes=dataset[target].unique().astype('str'), target_names = dataset[target].unique().astype('str'))
                
        # create two columns for side-by-side display
        col1, col2 = st.columns(2)

        # add explanation for confusion matrix
        with col1:
            with st.popover("What does this mean?"):
                st.write("""
                         A confusion matrix is a table that helps us visualize the four outcomes of classifications a model can make:
                         false positives, false negatives, true positives, and true negatives.
                         The rows indicate the true labels of the observations, and the columns indicate the predicted labels.
                         """)
                
        # add explanation for ROC and AUC
        with col2:
            with st.popover("What does this mean?"):
                st.write("""
                         The ROC Curve shows how much sensitivity we gain and how much specificity we lose by lowering the threshold of the probability
                         of belonging to a class required to classify an observation. Sensitivity is the true positive rate, and specificity is
                         the true negative rate. After we have the ROC curve, we calculate the AUC (the area underneath it).
                         A perfect model would have an AUC of 1, and an AUC of 0.5 would be as good as random guessing for binary classification.
                         """)

        # show decision tree
        st.subheader("Decision Tree")
        plot_decision_tree(model=dt_model, df=dataset, target=target,X_train=X_train)

        # option to view dataset information
        st.markdown("####")
        with st.expander("**View Dataset Information**"):
            if dataset_demo == "Breast Cancer":
                st.write("""
                         **Breast cancer wisconsin (diagnostic) dataset:** This is one of the toy datasets from the Python package scikit-learn.
                         It is used to predict whether a tumor is malignant or benign according to 30 predictive variables.
                         """)
            elif dataset_demo == "Iris":
                st.write("""
                        **Iris plants dataset:** This is one of the toy datasets from the Python package scikit-learn.
                        It is used to predict whether an Iris plant is from the species Setosa, Versicolour, or Virginica according to 4 predictive variables.
                        """)
            elif dataset_demo == "Wine":
                st.write("""
                        **Wine recognition dataset:** This is one of the toy datasets from the Python package scikit-learn.
                         It is used to predict whether a wine was manufactured by cultivator 0, 1, or 2 in Italy using 13 predictive variables.
                         """)
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### First 5 Rows of the Dataset")
                st.dataframe(dataset.head())
            with col2:
                st.write("#### Statistical Summary")
                st.dataframe(dataset.describe())

        # create space before common questions section
        st.markdown("####")

    # show best decision tree if user selects toggle
    else:
        
        # ensure scoring metric is selected
        if scoring_choice is None:
            st.markdown("#### :primary[Please select a scoring metric first.]")

        else:

            # find best tree and fit model
            y_pred, best_params, best_dtree = find_best_tree(X_train=X_train, y_train=y_train, X_test=X_test, scoring=scoring_metric)

            # create two columns for side-by-side display
            col1, col2 = st.columns(2)

            # best parameters
            with col1:
                st.subheader("Best Parameters")
                st.write(f"Using the scoring metric {scoring_choice}")
                st.dataframe(best_params)

            # classification report
            with col2:
                st.subheader("Classification Report")

                # evaluate model
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy:** {accuracy:.2f}")

                # display classification report
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).T
                st.dataframe(report_df)

            # create new columns for side-by-side display so they're vertically aligned
            col1, col2 = st.columns(2)

            # show confusion matrix on column 1 
            with col1:
                st.subheader("Confusion matrix")
                cm = confusion_matrix(y_test, y_pred)
                plot_confusion_matrix(cm)
            
            # show ROC and AUC plot on column 2
            with col2:
                st.subheader("ROC and AUC plot")
                plot_multiclass_roc(model=best_dtree, y_train= y_train,X_test=X_test, y_test=y_test, target_classes=dataset[target].unique().astype('str'), target_names = dataset[target].unique().astype('str'))
        
            # create two columns for side-by-side display
            col1, col2 = st.columns(2)

            # add explanation for confusion matrix
            with col1:
                with st.popover("What does this mean?"):
                    st.write("""
                            A confusion matrix is a table that helps us visualize the four outcomes of classifications a model can make:
                            false positives, false negatives, true positives, and true negatives.
                            The rows indicate the true labels of the observations, and the columns indicate the predicted labels.
                            """)
                    
            # add explanation for ROC and AUC
            with col2:
                with st.popover("What does this mean?"):
                    st.write("""
                            The ROC Curve shows how much sensitivity we gain and how much specificity we lose by lowering the threshold of the probability
                            of belonging to a class required to classify an observation. Sensitivity is the true positive rate, and specificity is
                            the true negative rate. After we have the ROC curve, we calculate the AUC (the area underneath it).
                            A perfect model would have an AUC of 1, and an AUC of 0.5 would be as good as random guessing for binary classification.
                            """)
        
            # show decision tree
            st.subheader("Decision Tree")
            plot_decision_tree(model=best_dtree, df=dataset, target=target,X_train=X_train)

            # option to view dataset information
            st.markdown("####")
            with st.expander("**View Dataset Information**"):
                if dataset_demo == "Breast Cancer":
                    st.write("""
                            **Breast cancer wisconsin (diagnostic) dataset:** This is one of the toy datasets from the Python package scikit-learn.
                            It is used to predict whether a tumor is malignant or benign according to 30 predictive variables.
                            """)
                elif dataset_demo == "Iris":
                    st.write("""
                            **Iris plants dataset:** This is one of the toy datasets from the Python package scikit-learn.
                            It is used to predict whether an Iris plant is from the species Setosa, Versicolour, or Virginica according to 4 predictive variables.
                            """)
                elif dataset_demo == "Wine":
                    st.write("""
                            **Wine recognition dataset:** This is one of the toy datasets from the Python package scikit-learn.
                            It is used to predict whether a wine was manufactured by cultivator 0, 1, or 2 in Italy using 13 predictive variables.
                            """)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### First 5 Rows of the Dataset")
                    st.dataframe(dataset.head())
                with col2:
                    st.write("#### Statistical Summary")
                    st.dataframe(dataset.describe())

        # create space before common questions section
        st.markdown("####")

# prompt user to select a dataset
elif dataset_upload is None and dataset_demo is None:
    st.markdown("#### :primary[Please select a dataset to start.]")

# prompt user to select a target variable for uploaded dataset
elif (dataset_upload or dataset_demo is not None) and target is None:
    st.markdown("#### :primary[Please select the target variable in your dataset. This should be a specific variable with the labels that you want the decision tree to use for classification.]")

# prompt user to select a criterion for the decision tree
elif criterion is None:
    st.markdown("#### :primary[Please select a criterion for the decision tree.]")

# create section for common questions
st.subheader("Common Questions")

# create columns for help buttons
col1, col2 = st.columns(2)

with col1:
    with st.expander("How do decision trees work?"):
        st.write("""Decision trees are supervised machine learning models that use labeled data for classification or regression. Here, we use them for **classification**.
                 These models traverse down the tree by asking questions at each **node** and deciding which **branch** to go through according to the answer, until they arrive at
                 a **leaf node**, where they assign the data point to a class.
                 """)
    with st.expander("What are hyperparameters?"):
        st.write("""
                 Hyperparameters are model settings chosen before the training process, and they determine how the model will learn and use the data.
                 **Hyperparameter tuning** consists of testing different parameters to find the best model according to the desired metric.
                 This application allows you to do this by manually changing the hyperparameters or by automatically finding the best decision tree.
                 """)
    with st.expander("How is the best decision tree determined?"):
        st.write("""
                 This app uses a method called **Grid Search Cross Validation** to find the combination of hyperparameters that yields the best decision tree according to your chosen scoring metric.
                 - **Grid search** allows us to test different combinations of hyperparameter values to see which one works the best.
                 - **Cross validation** splits the dataset into random groups (here, we use 5), holds one group as the test, and trains the model on the other ones.
                 This is repeated until each group is used as the test, and then the average performance is considered as the model performance for those parameters.
                 """)
    
with col2:
    with st.expander("What does each criterion mean?"):
        st.write("""
                 The criterion is the metric used to choose the best split (i.e. the best question to ask) at each node.
                 - **Gini Index:** A measure of the impurity of a dataset and corresponds to the probability of picking two elements of the same class from a set.
                 Usually the preferred metric for decision trees.
                 - **Entropy:** A measure of disorder based on information gain theory.
                 It evaluates the probability of picking the sequence corresponding to the initial ordered sequence of a set.
                 - **Log-Loss:** A measure of how close the predicted probability of an observation belonging to a class is to its actual class.
                 """)
    with st.expander("What does each hyperparameter mean?"):
        st.write("""
                 - **Maximum depth of tree:** The maximum amount of times nodes can be expanded into new branches.
                 - **Minimum samples to split:** The minimum number of samples that must be present in a node for it to split.
                 - **Minimum samples for leaf:** The minimum number of samples required for a node to be a leaf node (where observations are assigned to a class).
                 """)
    with st.expander("What do the scoring metrics mean?"):
        st.write("""
                 - **Accuracy:** The proportion of the data points that a model has classified with the correct label, i.e. the percentage of times the model makes a correct prediction.
                 - **F1 Score:** A harmonic mean of precision and recall.
                 - **Precision:** The proportion of data points classified as positive that were actually true positives, so TP / (TP + FP). For multiclass datasets, the precision for each class is averaged.
                 - **Recall:** The proportion of correct predictions out of the positive labels, i.e. the number of true positives divided by the total number of positives (TP / (TP + FN)).
                 For multiclass datasets, the recall for each class is averaged.
                 """)