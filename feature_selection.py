import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sheet_id = "1-eEohLGcKNl6f9l4JAN9rn5etY9o9xlL2AquExVb2kU"
sheet_name = "Sheet2"    
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url)

df = df.rename( columns={'Heat of formation': 'HoF',
                                       'Band gap': 'Eg',
                                       'Band gap (HSE06)': 'Eg(HSE)',
                                       'DOS at ef': 'DOS',
                                       'Energy': 'Et',
                                       'Fermi level': 'EFermi',
                                       'Speed of sound (x)': 'SosX',
                                       'Speed of sound (y)': 'SosY',
                                       'Work function (avg. if finite dipole)': 'Ew',
                                       'Static interband polarizability (x)': 'SePx',
                                       'Static interband polarizability (y)': 'SePy',
                                       'Dir. band gap': 'Dir Eg',
                                       'Stiffness tensor, 11-component': 'St11com',
                                       '2D plasma frequency (x)': '2DpfX',
                                       '2D plasma frequency (y)': '2DpfY',
                                       'Area of unit-cell': 'AuCell'
                                       })

#df = df.drop(columns=['Eg(HSE)','DOS','Dir Eg'])
target = 'AuCell'
no_of_features = 9

df_to_be_scaled = df.drop( columns= ['Formula', 'Type of Element'])
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_to_be_scaled)
df_scaled = pd.DataFrame(df_scaled, columns= df_to_be_scaled.columns)

y = df_scaled[target]
X = df_scaled.drop(columns = [target])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
trainindex = np.array(y_train.index.tolist())
testindex = np.array(y_test.index.tolist())

# Train a random forest classifier on the training data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
base_error = mean_absolute_error(y_test, y_pred)

# Initialize a dictionary to store the permutation feature importances
permutation_importances = {}

# Loop over each feature in the data
for feature in X.columns:
    # Create a copy of the test data with the feature shuffled
    X_test_permuted = X_test.copy()
    X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature])

    # Use the permuted test data to make predictions
    y_pred_permuted = model.predict(X_test_permuted)
    
    # Calculate the mean absolute error of the model on the permuted data
    error = mean_absolute_error(y_test, y_pred_permuted)
    
    # Store the difference in accuracy as the permutation feature importance
    permutation_importance = base_error - error
    permutation_importances[feature] = permutation_importance

# Convert the permutation importances to a pandas dataframe
permutation_importances = pd.DataFrame(list(permutation_importances.items()), columns=["Feature", "Importance"])

# Sort the permutation importances by the importance score
permutation_importances = permutation_importances.sort_values("Importance", ascending=True)

selected_features = permutation_importances.iloc[:no_of_features, 0]
selected_features.tolist()

feature_imp_list = permutation_importances.iloc[:,0].tolist()
feature_imp_value = -1*(permutation_importances.iloc[:,1])

fig, ax = plt.subplots()
ax.barh(feature_imp_list, feature_imp_value)

ax.grid(zorder=0,)
ax.set_axisbelow(True)
ax.set_facecolor('lightgrey')

title = 'Feature Importance for ' + target
plt.xlabel("Importance", fontdict= {'weight': 'bold'}, size= 25)
plt.title(title, fontdict= {'weight': 'bold'}, size= 25)
xticks = ax.get_xticks()
yticks = ax.get_yticks()
plt.xticks(xticks, weight= 'bold', size= 20)
plt.yticks(yticks, weight= 'bold', size= 20)

fig.set_dpi(300)

plt.show()

datagg = df.loc[:,target]

fig2, ax2 = plt.subplots()

# create a histogram of the data
min_value = df[target].min()
max_value = df[target].max()
ax2.hist(datagg, bins=30, density=True, alpha=0.5, color='b')
# create a KDE of the data
from scipy.stats import gaussian_kde
kde = gaussian_kde(datagg)
x = np.linspace(min_value, max_value, 100)
ax2.plot(x, kde(x), linewidth=2, color='r')

# add a title and labels
title = 'Distribution of ' + target
plt.title(title, fontdict= {'weight': 'bold', 'size': '25'})
plt.xlabel('Value', fontdict= {'weight': 'bold', 'size': '20'})
plt.ylabel('Density', fontdict= {'weight': 'bold', 'size': '20'}, labelpad= 10)
xticks = ax2.get_xticks()[1:-1]
yticks = ax2.get_yticks()[1:-1]
plt.xticks(xticks, weight= 'bold', size= 20)
plt.yticks(yticks, weight= 'bold', size= 20)
fig2.gca().set_facecolor('#f0f0f0')

fig2.set_dpi(300)

plt.show()

selected_features = selected_features.tolist()
print(selected_features)