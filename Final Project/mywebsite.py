from flask import Flask, render_template, request
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
import pickle
import numpy as np
from sklearn import decomposition
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def homepage():
    data = pd.read_csv('cleaned_data.csv', dtype=np.float)
    fig = px.scatter(data,
                     x=np.linspace(1, 295, 295, endpoint=True),
                     y='serum_creatinine', labels={"x": 'patient index', "y": 'mg/dL'}, title='Serum_creatinine'
                     )
    fig.add_scatter(x=np.linspace(1, 295, 295, endpoint=True),
                    y=[1.2] * 295,
                    mode='lines',
                    hoverinfo='none',
                    line=dict(dash='dot', color='red'),
                    name='1.2mg/dL')
    fig.update_layout(yaxis=dict(title='mg/dL'))
    figure_json1 = fig.to_json()

    fig = tls.make_subplots(rows=1, cols=2)
    # Add first histogram
    fig.add_trace(go.Histogram(x=data.age, name='Age Histogram',nbinsx=16), row=1, col=1)

    # Add second histogram
    fig.add_trace(go.Histogram(x=data.ejection_fraction, name='ejection_fraction Histogram', nbinsx=16), row=1, col=2)
    fig.add_scatter(x=[50] * 100, y=np.linspace(1,100,100,endpoint=True),
                    mode='lines',
                    hoverinfo='none',
                    line=dict(dash='dot', color='blue'),
                    name='50%',row=1,col=2)
    fig.add_scatter(x=[70] * 100, y=np.linspace(1, 100, 100, endpoint=True),
                    mode='lines',
                    hoverinfo='none',
                    line=dict(dash='dot', color='gray'),
                    name='70%', row=1, col=2)
    # Define subplot titles and axis labels
    fig.update_layout(
        title='Age and ejection_fraction Histograms',
        xaxis=dict(title='age'),
        yaxis=dict(title='frequency'),
    )
    fig.update_xaxes(title_text='ejection_fraction', row=1, col=2)
    figure_json2 = fig.to_json()

    fig3 = px.scatter(data,
                     x=np.linspace(1, 295, 295, endpoint=True),
                     y='serum_sodium', labels={"x": 'patient index', "y": 'mmol/L'}, title='Serum_sodium'
                     )
    fig3.add_scatter(x=np.linspace(1, 295, 295, endpoint=True),
                    y=[135] * 295,
                    mode='lines',
                    hoverinfo='none',
                    line=dict(dash='dot', color='red'),
                    name='135mmol/L')
    fig3.add_scatter(x=np.linspace(1, 295, 295, endpoint=True),
                     y=[145] * 295,
                     mode='lines',
                     hoverinfo='none',
                     line=dict(dash='dot', color='gray'),
                     name='145mmol/L')
    fig3.update_layout(yaxis=dict(title='mmol/L'))
    figure_json3 = fig3.to_json()

    fig4 = px.pie(data, names="DEATH_EVENT", title='Death event')
    figure_json4 = fig4.to_json()

    return render_template('index.html', figure_json1=figure_json1, figure_json2=figure_json2, figure_json3=figure_json3, figure_json4=figure_json4)

@app.route('/interaction',methods=['GET','POST'])
def getvalues():
    if request.method == 'GET':
        return render_template('interact.html')
    elif request.method == 'POST':
        age = request.form.get('age')
        anaemia = request.form.get('anaemia')
        high_blood_pressure = request.form.get('high_blood_pressure')
        creatinine_phosphokinase = request.form.get('creatinine_phosphokinase')
        diabetes = request.form.get('diabetes')
        ejection_fraction = request.form.get('ejection_fraction')
        platelets = request.form.get('platelets')
        sex = request.form.get('sex')
        serum_creatinine = request.form.get('serum_creatinine')
        serum_sodium = request.form.get('serum_sodium')
        smoking = request.form.get('smoking')
        time = request.form.get('time')

        input_vector=np.array([age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,
                      high_blood_pressure,platelets,serum_creatinine,serum_sodium,
                      sex,smoking,time]).reshape(1,-1)


        # Standardization
        # mean and std of the training set
        mean = np.array([6.08042087e01, 4.22330097e-01, 5.56077670e+02, 4.22330097e-01,
                3.77669903e+01, 3.30097087e-01, 2.70237924e+05, 1.36713592e+00,
                1.36728155e+02, 6.65048544e-01, 3.39805825e-01, 1.30956311e+02]).reshape(1,-1)
        std = np.array([1.15888744e+01, 4.93930548e-01, 8.38148934e+02, 4.93930548e-01,
                        1.15667786e+01, 4.70247807e-01, 1.00616641e+05, 8.90432998e-01,
                        4.52113521e+00, 4.71973493e-01, 4.73643142e-01, 7.64563651e+01]).reshape(1,-1)

        # If there are any missing input values, take default values instead
        for i in range(len(input_vector[0])):
            if input_vector[0][i] in [None,'']:
                input_vector[0][i] = mean[0][i]

        input_vector_std=(np.array(input_vector, dtype=np.float)-mean)/std
        print(input_vector_std.shape)

        filenames = ['logistic.sav', 'ridge.sav', 'randomforest.sav', ]
        model_name = request.form.get('model')
        if model_name=='logistic':
            filename = filenames[0]
            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.predict(input_vector_std)
            prob = np.max(loaded_model.predict_proba(input_vector_std))*100
            return render_template('prediction.html', result=result, prob=prob)
        elif model_name == 'ridge':
            filename = filenames[1]
            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.predict(input_vector_std)
            return render_template('prediction.html', result=result)
        elif model_name == 'randomforest':
            filename = filenames[2]
            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.predict(input_vector_std)
            prob = np.max(loaded_model.predict_proba(input_vector_std))*100
            return render_template('prediction.html', result=result, prob=prob)
        else:
            return 'You need to choose a specific model'

@app.route('/pca',methods=['GET','POST'])
def pcainter():
    if request.method == 'GET':
        showpca = False
        return render_template('pcainteract.html', showpca=showpca)
    elif request.method == 'POST':
        pca_n = request.form.get('pcacom')
        pca_graphx = request.form.get('pcagraphx')
        pca_graphy = request.form.get('pcagraphy')
        data = pd.read_csv('cleaned_data.csv',dtype=np.float)
        STDS = StandardScaler()
        X = data.drop(['DEATH_EVENT'],axis=1)
        STDS.fit(X)
        X = STDS.transform(X)
        print(X.shape)
        y = data['DEATH_EVENT']
        pca = decomposition.PCA(n_components=int(pca_n))
        data_reduced = pca.fit_transform(X)
        pc0 = data_reduced[:, int(pca_graphx)]
        pc1 = data_reduced[:, int(pca_graphy)]

        plt.figure(figsize=(8, 6))
        for c in np.unique(y):
            i = np.where(np.array(y) == c)
            print('number of points in class %s' % c, np.array(pc0)[i].shape)
            plt.scatter(np.array(pc0)[i], np.array(pc1)[i], label=c)
        plt.xlabel('PC%s' %pca_graphx)
        plt.ylabel('PC%s' %pca_graphy)
        plt.title('PC%s vs PC%s' % (pca_graphx , pca_graphy))
        plt.legend()
        plt.savefig("static/interactpca.png")
        explained = pca.explained_variance_ratio_
        total_info = np.round(np.sum(explained)*100,2)
        showpca = True
        return render_template('pcainteract.html', explained=explained, showpca=showpca, total_info=total_info, pca_n=pca_n)

@app.route('/correlation')
def correlation():
    data = pd.read_csv('cleaned_data.csv', dtype=np.float)
    plt.figure(figsize=(18, 18))
    sns.heatmap(np.round(data.corr(), 2), annot=True)
    plt.savefig('static/corr.png', bbox_inches='tight')
    return render_template('correlation.html')

if __name__ =="__main__":
    app.run(debug=True)