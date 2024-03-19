
#Import dependencies
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from ucimlrepo import fetch_ucirepo 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost.sklearn import XGBRegressor
pd.set_option('display.max_columns',None)

#Load our data
df = fetch_ucirepo(id=189) 


# In[3]:


#Split our dataset
X = df.data.features 
y = df.data.targets 



#since the dataset already describes the objective columns we exclude them from our predictor space (p.s.)
tgt = ['motor_UPDRS','total_UPDRS']
var = [col for col in X if col not in ['subject#']+tgt]



#source also refers there are no missings, let's see a brief description of each column on our p.s.
X[var].describe(percentiles=np.arange(0,1,.2))



#by definition we are expecting to have positive numbers on test_time since it contains the days the subject was recruited, let's erase the negative ones



X = X.loc[X['test_time']>=0]

y = y.loc[y.index.isin(X.index)]

#since we don't have enough context to identify outliers let's just plot the distribution of variance and standard deviation to understand more our variables



#calculate them:
variances = []
deviations = []
for v in var:
    variances.append(np.var(X[v]))
    deviations.append(np.std(X[v]))
#we temporally name aux to help our code execution
aux = pd.DataFrame(zip(var,variances,deviations), columns=['feature','variance','deviation'])





aux.plot(kind='line',backend='plotly', x='feature',y=['variance','deviation'])


#as we can see there might be an outlier for the test_time feature, let's dive in



import plotly.figure_factory as ff
fig = ff.create_distplot([X['test_time']], group_labels=['test_time'], colors=['blue'], show_rug = False)
fig.update_layout(
    title='KDE Plot for test_time',
    xaxis=dict(title='Values'),
    yaxis=dict(title='Density')
)
fig.show()


#we can see it doesn't follow a normal distribution, due to the lack of context we won't be filtering by a value on this column for now


#let's plot the distribution for the other variables to have a quick sense of what we're dealing with


var.remove('test_time')


fig = ff.create_distplot([X[v] for v in var], group_labels=var, histnorm='probability density')
fig.update_layout(
    title='KDE Plot for Variables',
    xaxis=dict(title='Values'),
    yaxis=dict(title='Density')
)
fig.show()


#as we can see some of our features follow a normal or a log-normal distribution, we will make this log-normal variables follow a normal distribution 


log_norms = [col for col in X if col.startswith(('Shimmer','Jitter'))]


Xn = X.copy()


for v in log_norms:
    Xn[v], lam = boxcox(Xn[v])


#Boxcox is a powerful transformation that aims to stabilize variance. It works by estimating a lambda parameter maximizing its likelihood function.



fig = ff.create_distplot([Xn[v] for v in var], group_labels=var, histnorm='probability density')
fig.update_layout(
    title='KDE Plot for Variables',
    xaxis=dict(title='Values'),
    yaxis=dict(title='Density')
)
fig.show()




#As we can see this clearly works 



#Now that we have a deeper understanding about our p.s. let's select the variables by using SelectKBest with the f_regression scoring method
#this method works by computing the F-value between each feature on our p.s. to the target, then we can select the highest values to select features



#for this we need to split into train valid and dev sets



Xt, Xv, yt, yv=train_test_split(Xn, y[[tgt[0]]], test_size=.2, random_state=17)


#now we have this let's select our 6 best features


sk = SelectKBest(k=6, score_func=f_regression)
sk.fit(Xt[var], yt)
best = [a for a,b in zip(var,sk.get_support()) if b]


pd.DataFrame(zip(var,sk.scores_),
             columns=['var','score']).set_index('var').sort_values(by='score',
                                                                   ascending=False).plot(kind='bar'
                                                                                          ,backend='plotly')



#we can see that age is our strongest variable for our first target variable, lets fit a model



lr = LinearRegression()



lr.fit(Xt[best],yt)


print(mean_absolute_error(yt, lr.predict(Xt[best])))
print(mean_absolute_error(yv, lr.predict(Xv[best])))
print(r2_score(yt, lr.predict(Xt[best])))
print(r2_score(yv, lr.predict(Xv[best])))


y[[tgt[0]]].describe(percentiles=np.arange(0,1,.1))


#we can see there's no underfitting or overfitting and our error is not that bad let's try a more complex model


xgb = XGBRegressor()


xgb.fit(Xt[best],yt)


print(mean_absolute_error(yt, xgb.predict(Xt[best])))
print(mean_absolute_error(yv, xgb.predict(Xv[best])))
print(r2_score(yt, xgb.predict(Xt[best])))
print(r2_score(yv, xgb.predict(Xv[best])))

#we can see a much better score but an overfitting is also visible



