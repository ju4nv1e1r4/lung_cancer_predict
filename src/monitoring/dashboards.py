import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


# Model training
logged_model = 'src/models/mlartifacts/210001136261627048/a4c15c5934734298994e3ffe71cfdaf5/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)

df = pd.read_csv('data/processed/preprocessed.csv')

X = df.drop(columns=['lung_cancer', 'gender', 'age', 'gen_flag', 'generation'])
y = df['lung_cancer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Model metrics
accuracy_test = accuracy_score(y_test, y_pred_test)
f1score_test = f1_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
cm_test = confusion_matrix(y_test, y_pred_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
f1score_train = f1_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train)
recall_train = recall_score(y_train, y_pred_train)
cm_train = confusion_matrix(y_train, y_pred_train)

df_metrics = pd.DataFrame({'Metrics': ['Accuracy Score Test', 'F1 Score Test',
                                       'Precision Score Test', 'Recall Score Test',
                                       'Accuracy Score Train', 'F1 Score Train',
                                       'Precision Score Train', 'Recall Score Train'],
                           'Score':   [accuracy_test, f1score_test, precision_test, 
                                       recall_test, accuracy_train, f1score_train, 
                                       precision_train, recall_train]})

# Dashboard
st.title('Metrics Monitor')

def bar_metrics_plot():
    st.subheader('Metrics')
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_metrics['Metrics'],
        y=df_metrics['Score'],
        name='Metrics',
        marker_color='lightblue',
    ))

    fig.update_layout(barmode='group', title='Metrics')
    st.plotly_chart(fig)


def conf_matrix_test_plot():
    st.subheader('Confusion Matrix - Test Side')
    fig = px.imshow(cm_test,
                    text_auto=True)
    
    st.plotly_chart(fig)


def conf_matrix_train_plot():
    st.subheader('Confusion Matrix - Train Side')
    fig = px.imshow(cm_train,
                    text_auto=True)
    
    st.plotly_chart(fig)


def trp_fpr():
    tp,  fn , fp, tn = cm_test[0, 0], cm_test[0, 1], cm_test[1, 0], cm_test[1, 1]

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    df_tpr_fpr = pd.DataFrame({'Type': ['TPR', 'FPR'],
                               'Rate': [tpr, fpr]})

    st.subheader('True Positive Rate x False Positive Rate')
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_tpr_fpr['Type'],
        y=df_tpr_fpr['Rate'],
        name='TPR x FPR',
        marker_color='lightgreen'
    ))

    fig.update_layout(barmode='group', title='Metrics')
    st.plotly_chart(fig)


def class_report_test():
    st.subheader('Classification Report - Test Side')
    classif_report = classification_report(y_test ,y_pred_test, output_dict=True)
    df_report = pd.DataFrame(classif_report).transpose()
    st.write(df_report)


def class_report_train():
    st.subheader('Classification Report - Train Side')
    classif_report = classification_report(y_train ,y_pred_train, output_dict=True)
    df_report = pd.DataFrame(classif_report).transpose()
    st.write(df_report)


st.sidebar.title('Choose here your visualization:')
option = st.sidebar.selectbox(
    'Choose Visualization:',
    ['Metrics', 'Confusion Matrix - Test', 'Confusion Matrix - Train',
     'True Positive Rate x False Positive Rate', 'Classification Report - Test',
     'Classification Report - Train']
)

if option == 'Metrics':
    bar_metrics_plot()
elif option == 'Confusion Matrix - Test':
    conf_matrix_test_plot()
elif option == 'Confusion Matrix - Train':
    conf_matrix_train_plot()
elif option == 'True Positive Rate x False Positive Rate':
    trp_fpr()
elif option == 'Classification Report - Test':
    class_report_test()
elif option == 'Classification Report - Train':
    class_report_train()
