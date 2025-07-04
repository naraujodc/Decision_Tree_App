# Decision Tree App
![Python](https://img.shields.io/badge/-Python-ffe873?style=flat&logo=python)&nbsp;
![Streamlit](https://img.shields.io/badge/Streamlit-ececec?style=flat&logo=streamlit)&nbsp;
![Scikitlearn](https://img.shields.io/badge/scikit_learn-101e27?logo=scikitlearn)&nbsp;
![NumPy](https://img.shields.io/badge/numpy%20-%23013243.svg?&style=flat&logo=numpy&logoColor=white)&nbsp;
![Pandas](https://img.shields.io/badge/pandas%20-%23150458.svg?&style=flat&logo=pandas&logoColor=white)&nbsp;
![Seaborn](https://img.shields.io/badge/Seaborn-79b6bc)&nbsp;
![Matplotlib](https://img.shields.io/badge/Matplotlib-1e3f5a)&nbsp;
![Graphviz](https://img.shields.io/badge/graphviz-5d81a3)&nbsp;

## Table of Contents
[1. Project Overview (With Visual Examples)](#project-overview-with-visual-examples)\
[2. Instructions for Use](#instructions-for-use)\
[3. Files](#files)\
[4. Datasets Description](#datasets-description)\
[5. References](#references)

## Project Overview (With Visual Examples)

<p align="center">
<a href="https://decision-tree-tuning.streamlit.app"><img alt="Decision Tree Hyperparameter Tuning App" src="https://img.shields.io/badge/Decision_Tree_Hyperparameter_Tuning_App-ddf2d1?style=for-the-badge"/></a> &nbsp;
</p>

This project is an interactive app that invites the user to explore the effects of hyperparameters on decision tree machine learning models. The user is able to:
- Upload their own dataset or choose from three demo datasets;
- Use dropdown buttons and sliders to change four hyperparameters (criterion, maximum depth of tree, minimum number of samples to split, and minimum number of samples for a leaf);
- Automatically find the best decision tree for the data according to a scoring metric of their choice;
- Visualize the **Classification Report**, **Confusion Matrix**, **ROC and AUC Plot**, and the **Decision Tree Structure** for each decision tree;
- View explanations about each aspect of the decision tree, including the hyperparameters and the metrics & visuals cited above;
- View the first observations of their chosen dataset and its summary statistics, as well as short descriptions of the demo datasets.

The purpose of this project is to provide an interactive experience and enhance the user's understanding of decision trees.
My goal was to make the app as self-contained as possible, so there are answers to common questions that may show up and explanations about every aspect of the decision tree models within the app.
This project showcases my knowledge of this powerful machine learning model and demonstrates my ability to interpret and tune hyperparameters to find the best model under different contexts.

### Behind the App
Many computations happen in the background to ensure that the users can navigate the app smoothly and without error.
I highly recommend checking the code in this repository to see some of the strategies I used to guarantee that the code runs without errors independently of the order in which users interact with the buttons.
<p align="center">
<img src="https://github.com/naraujodc/Decision_Tree_App/blob/main/images/decision-tree-app-initial-page.png">
Screenshot of the initial page when a user opens the app
</p>

#### Data Preprocessing
To make sure each dataset was ready for machine learning, I created a function to preprocess any user-uploaded datasets.
Since decision trees can naturally handle missing values, I decided not to delete or input them. This ensures I would not delete a lot of data from datasets with many missing values.
I had to encode categorical variables in order to use `scikit-learn`'s `DecisionTreeClassifier`. In addition, I separated the feature and target variables according to the target variable indicated by the user.
```
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
```
After applying this function, the app separates the dataset into training and testing sets and trains the decision tree model according to the hyperparameters chosen by the user.

#### Best Tree Finder
The user also has the option to find the best set of decision tree hyperparameters for their dataset according to their chosen scoring metric (accuracy, f1 score, precision, or recall).
This is computed using **grid search cross validation**.
- **Grid search** allows us to test different combinations of hyperparameter values to see which one works the best.
- **Cross validation** splits the dataset into random groups (here, we use 5), holds one group as the test, and trains the model on the other ones.
This is repeated until each group is used as the test, and then the average performance is considered as the model performance for those parameters.
```
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
```

## Instructions for Use
The app has been deployed and can be accessed with the URL below. To use the app, simply visit the following link:
<p align="center">
<a href="https://decision-tree-tuning.streamlit.app"><img alt="Decision Tree Hyperparameter Tuning App" src="https://img.shields.io/badge/Decision_Tree_Hyperparameter_Tuning_App-ddf2d1?style=for-the-badge"/></a> &nbsp;
</p>

If you want to download and run the code yourself, follow the instructions below:
1. Install Anaconda.
2. Clone this repository and cd into it.
```
git clone https://github.com/naraujodc/Araujo_Data_Science_Portfolio

cd MLStreamlitApp
```
3. In a new environment, install the dependencies.
```
conda activate new_env

pip install -r requirements.txt
```
4. In the terminal, make sure you are in `MLStreamlitApp`. If not, cd into it.
5. In the terminal, run the app on Streamlit.
```
streamlit run decision-tree.py
```

## Files
- `decision-tree.py` &rarr; This Python file contains all the code for the app.
- `requirements.txt` &rarr; List of required Python libraries.
- `README.md` &rarr; Project documentation.
- `images` &rarr; This directory stores the images used in the documentation.
- `.streamlit` &rarr; This directory stores `config.toml`, a file containing the custom theme I made for the app.

## Datasets Description
The demo datasets provided in the app are [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) from `scikit-learn`.
- **Breast cancer wisconsin (diagnostic) dataset:** Used to predict whether a tumor is malignant or benign according to 30 (numerical) predictive variables.
- **Iris plants dataset:** Used to predict whether an Iris plant is from the species Setosa, Versicolour, or Virginica according to 4 (numerical) predictive variables.
-  **Wine recognition dataset:** Used to predict whether a wine was manufactured by cultivator 0, 1, or 2 in Italy using 13 (numerical) predictive variables.

## References
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Stack Overflow](https://stackoverflow.com/questions)
- [Gemini](https://gemini.google.com/app) (Used for assistance with the `plot_multiclass_roc` function)
