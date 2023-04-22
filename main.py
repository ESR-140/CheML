import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_selection_results import get_features
from models import rmse, GBR, MLR, LASSO, RidgeReg, KRR, GPR, RFR


sheet_name = "Sheet2"
sheet_id = "1-eEohLGcKNl6f9l4JAN9rn5etY9o9xlL2AquExVb2kU"
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

target = 'EFermi'
model = GBR()
#selected_features = ['Eg']
selected_features = get_features(target, all_features='Yes')
selected_features.append(target)

df_to_be_scaled = df.drop( columns= ['Formula', 'Type of Element'])
df_to_be_scaled = df_to_be_scaled.loc[:,selected_features]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_to_be_scaled)
df_scaled = pd.DataFrame(df_scaled, columns= df_to_be_scaled.columns)



y = df_scaled[target]
X = df_scaled.drop(columns = [target])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
trainindex = np.array(y_train.index.tolist())
testindex = np.array(y_test.index.tolist())

fit = model.fit(X_train, y_train)

y_pred_test = model.predict(fit, X_test)
y_pred_train = model.predict(fit, X_train)

df_scaled_copy = df_scaled.copy()
df_scaled_copy.loc[trainindex, target] = y_pred_train
df_scaled_copy.loc[testindex, target] = y_pred_test
df_inverse = scaler.inverse_transform(df_scaled_copy)
column = df_scaled.columns.tolist()
df_inverse = pd.DataFrame(df_inverse, columns= column)

y_pred_test_original = df_inverse.loc[testindex, target]
y_pred_train_original = df_inverse.loc[trainindex, target]
y_test_original = df.loc[testindex, target]
y_train_original = df.loc[trainindex, target]

# Evaluate the mean average error
mae_test = mean_absolute_error(y_test_original, y_pred_test_original)
mae_train = mean_absolute_error(y_train_original, y_pred_train_original)
rmse_test = rmse(y_test_original, y_pred_test_original)
rmse_train = rmse(y_train_original, y_pred_train_original)
print("{:.3f} \t {:.3f} \t {:.3f} \t {:.3f}".format(mae_train, mae_test, rmse_train, rmse_test))

target_index = df.columns.get_loc(target)
metal_indices = df[df['Type of Element'] == 'Metal'].index.tolist()
NonMetal_indices = df[df['Type of Element'] == 'Non-metal'].index.tolist()
target_metals = df.iloc[metal_indices,target_index]
target_nonmetals = df.iloc[NonMetal_indices,target_index]
y_test_original_metal_index = list(set(metal_indices) & set(testindex))
y_test_original_metal = y_test_original[y_test_original_metal_index]
y_pred_test_original_metal = y_pred_test_original[y_test_original_metal_index]
y_test_original_nonmetal_index = list(set(NonMetal_indices) & set(testindex))
y_test_original_nonmetal = y_test_original[y_test_original_nonmetal_index]
y_pred_test_original_nonmetal = y_pred_test_original[y_test_original_nonmetal_index]

fig, ax = plt.subplots()

# Plot the data
min_value = df[target].min()
max_value = df[target].max()
straightline = np.linspace(min_value, max_value, 1000)

line1 = ax.plot(straightline, straightline, '-r', '--')
line2 = ax.scatter(y_test_original_nonmetal, y_pred_test_original_nonmetal, marker="o", color="green")
line3 = ax.scatter(y_test_original_metal, y_pred_test_original_metal, marker="^", color="blue")

title = "Actual vs ML result for " + target #+ " with Eg only"


fig.tight_layout()
#fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

xticks = ax.get_xticks()[1:-1]
plt.xticks(xticks, weight= 'bold', size= 20)
ax.set_yticklabels(xticks)
plt.yticks(xticks, weight= 'bold', size= 20)

ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_facecolor('lightgrey')


ax.set_title(title, fontdict= {'weight': 'bold', 'size': 25})
ax.set_xlabel("Actual Result", size=20, fontdict= {'weight': 'bold'})
ax.set_ylabel("ML Result", size=20, fontdict= {'weight': 'bold'})
ax.legend([line2, line3], ["Non-Metal", "Metal"])

plt.tight_layout()

fig.set_dpi(1000)

#plt.show()