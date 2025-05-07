import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Activation,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet
from prince import MCA

scaler = MinMaxScaler()


dd = pd.read_excel("Location_data.xlsx")

df_new = pd.read_excel("New_data_for_predictions.xlsx")
df_new = df_new.set_index("Date")
df_new.plot(figsize=(20, 10))

day = 12
k = 0
array = []
array_temp = []
train_data = []
train_labels = []

for i in range(len(df_new)):
    array_temp.append(df_new.iloc[i]['Cases'])

array_temp = np.array(array_temp).reshape(-1,1)
array_temp = scaler.fit_transform(array_temp)
array_temp = array_temp.tolist()

for i in array_temp:
    array.append(i[0])

for i in range(len(array)):
    try:
        train_data.append(array[k:day+k])
        train_labels.append([array[day+k]])
        k+=1
    except:
        break
length = max(map(len, train_data))
train_data=np.array([xi+[None]*(length-len(xi)) for xi in train_data]).astype('float32')

length = max(map(len, train_labels))
train_labels = np.array([xi+[None]*(length-len(xi)) for xi in train_labels]).astype('float32')


train_data = train_data[:len(train_labels)]
train_data = np.expand_dims(train_data,1)


model = Sequential()
model.add(LSTM(256, input_shape=(1, day)))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(day, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='relu'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

E = 1000
callback = EarlyStopping(monitor='loss', mode='min', patience=20)
H = model.fit(train_data,train_labels,epochs=E, verbose=1, callbacks=[callback])

loss = H.history['loss']
epochs = range(0, len(loss))
plt.figure(figsize=(15,5))
plt.plot(epochs,loss)


preds = scaler.inverse_transform(model.predict(train_data))
plt.figure(figsize=(30,10))
plt.plot(preds,label='our predictions')
plt.plot(scaler.inverse_transform(train_labels),label='real values')
plt.legend()

days_to_predict = 12
seed = array[-day:]
for _ in range(days_to_predict):
    current_days = seed[-day:]
    current_days = np.squeeze(current_days)
    current_days = np.expand_dims(current_days,0)
    current_days = np.expand_dims(current_days,0)
    pred = model.predict(current_days)
    seed = np.append(seed,pred)

upcoming_days_prediction = scaler.inverse_transform(seed[-days_to_predict:].reshape(-1,1))

# Adding real values and predicted values together
arr_without_pred = scaler.inverse_transform(train_labels)
arr_pred = scaler.inverse_transform(seed[-days_to_predict:].reshape(-1,1))
arr_with_pred = np.concatenate((arr_without_pred, arr_pred))

plt.figure(figsize=(30,10))
plt.plot(arr_with_pred)

total_new_cases = 0
for i in upcoming_days_prediction:
    total_new_cases += i[0]
    print(i[0])
print(total_new_cases)

normalized_predictions = upcoming_days_prediction/ np.sum(upcoming_days_prediction)
actual_data = [0.001762257, 0.000516846, 0.000345602, 0.000445235, 0.003225616, 0.018544176, 0.136568705, 0.224099334, 0.247802627, 0.211000719, 0.120543373, 0.035145511]
#actual_data = [566, 166, 111, 143, 1036, 5956, 43863, 71976, 79589, 67769, 38716, 11288]
plt.figure(figsize=(30,10))
plt.plot(normalized_predictions,label='our predictions')
plt.plot(actual_data,label='real values')
plt.legend()
print(normalized_predictions)


df_new = pd.read_excel("New_data_for_predictions.xlsx")
df_new['Cases'] = df_new['Cases'].apply(lambda x: max(0, x))
df_new['Date'] = pd.to_datetime(df_new['Date'])
df = df_new.rename(columns={'Date': 'ds', 'Cases': 'y'})
model = Prophet(yearly_seasonality = True, seasonality_prior_scale=0.5)
model.fit(df)
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)
forecast_2023 = forecast[(forecast['ds'].dt.year == 2023)][['ds', 'yhat']]
print(forecast_2023)

prediction_list = forecast_2023['yhat'].to_list()
total_num = np.sum(prediction_list)
for i in range(len(prediction_list)):
    prediction_list[i] = prediction_list[i]/total_num
    
actual_data = [0.001762257, 0.000516846, 0.000345602, 0.000445235, 0.003225616, 0.018544176, 0.136568705, 0.224099334, 0.247802627, 0.211000719, 0.120543373, 0.035145511]
#actual_data = [566, 166, 111, 143, 1036, 5956, 43863, 71976, 79589, 67769, 38716, 11288]
print(prediction_list)
plt.figure(figsize=(30,10))
plt.plot(prediction_list,label='our predictions')
plt.plot(actual_data,label='real values')
plt.legend()

df_location_data = pd.read_excel("Location_dataset.xlsx")
all_district = df_location_data.columns
all_district_list = all_district[2:].tolist()

district_wise_weight = {}
full_bangladesh_count = np.sum(df_location_data['Bangladesh'])
for i in all_district_list:
    district_wise_weight[i] = np.sum(df_location_data[i])/full_bangladesh_count
    

total_sum = 0
for value in district_wise_weight.values():
    total_sum += value
print("Total sum using loop:", total_sum)


dengue_data_final = pd.read_excel("Dengue Symptoms Dataset.xlsx")


def reduce_with_mca(df, col1, col2, new_col_name):
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Columns '{}' and '{}' must exist in the DataFrame.".format(col1, col2))

    filtered_df = df[(~df[col1].isna()) | (~df[col2].isna())]
    filtered_df[col1] = filtered_df[col1].fillna(filtered_df[col2])
    filtered_df[col2] = filtered_df[col2].fillna(filtered_df[col1])
    
    mca = MCA(n_components=1, copy=True)
    mca = mca.fit(filtered_df[[col1, col2]])
    transformed_data = mca.transform(df[[col1, col2]])
    transformed_df = pd.DataFrame(transformed_data)
    df[new_col_name] = transformed_df
    return df


def reduce_with_mca_three(df, col1, col2, col3, new_col_name):
    if col1 not in df.columns or col2 not in df.columns or col3 not in df.columns:
        raise ValueError("Columns '{}' and '{}' and '{}' must exist in the DataFrame.".format(col1, col2, col3))

    combined_df = df[[col1, col2, col3]].copy()
    combined_df.dropna(inplace=True)
    mca = MCA(n_components=1)
    mca.fit(combined_df)
    transformed_data = mca.transform(df[[col1, col2, col3]])
    transformed_df = pd.DataFrame(transformed_data)
    df[new_col_name] = transformed_df

    return df

dengue_data_final = reduce_with_mca_three(dengue_data_final.copy(), "Headaches", "Pain behind the eyes", "Fatigue", "Headache_EyePain_Fatigue_MCA")
dengue_data_final = reduce_with_mca_three(dengue_data_final.copy(), "Muscle pain","Joint pain", "Belly pain", "Muscle_Joint_Belly_Pain_MCA")
dengue_data_final=  reduce_with_mca(dengue_data_final.copy(), "Metallic taste in the mouth","Appetite loss", "Appetite_Loss/Metalic_Taste_MCA")
dengue_data_final = reduce_with_mca(dengue_data_final.copy(), "Nausea/vomiting","Diarrhea", "Nausea/vomiting_Diarrhea_MCA")
dengue_data_final = reduce_with_mca(dengue_data_final.copy(), "Being very thirsty","Rapid Breathing", "Thirsty/Rapid_Breathing_MCA")


age_bins = [0, 18, 29, 39, 59, 85]
age_labels = ['0-18', '19-29', '30-39', '40-59', '60+']
dengue_data_final['Age_Group'] = pd.cut(dengue_data_final['Age'], bins=age_bins, labels=age_labels, right=False)

temperature_bins = [96, 99, 101, 103, 105, 107]
temperature_labels = ['Low', 'Noraml', 'High1', 'High2', 'High3']
dengue_data_final['Temperature_Group'] = pd.cut(dengue_data_final['Maximum body temperature'], bins=temperature_bins, labels=temperature_labels, right=False)

df_final_dengue_data = dengue_data_final[["Gender", "Age_Group", "Temperature_Group", "Headache_EyePain_Fatigue_MCA", 
                                         "Muscle_Joint_Belly_Pain_MCA", "Appetite_Loss/Metalic_Taste_MCA", "Nausea/vomiting_Diarrhea_MCA", 
                                         "Rash", "Blood in vomit or stool", "Bleeding gums or nose", "Swollen glands", "Dengue Fever"]]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="most_frequent")
df_final_dengue_data = pd.DataFrame(imputer.fit_transform(df_final_dengue_data), columns=df_final_dengue_data.columns)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


label_encoders = {}
categorical_cols = ["Gender", "Age_Group", "Temperature_Group", "Rash", "Blood in vomit or stool", "Bleeding gums or nose", "Swollen glands", "Dengue Fever"]
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df_final_dengue_data[col] = label_encoders[col].fit_transform(df_final_dengue_data[col])


numeric_cols = ["Headache_EyePain_Fatigue_MCA", "Muscle_Joint_Belly_Pain_MCA", "Appetite_Loss/Metalic_Taste_MCA", "Nausea/vomiting_Diarrhea_MCA"]
df_final_dengue_data[numeric_cols] = df_final_dengue_data[numeric_cols].astype(float)

X = df_final_dengue_data.drop("Dengue Fever", axis=1)
y = df_final_dengue_data["Dengue Fever"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
print("Report of Random Forest")
print(classification_report(y_test, y_pred))

feature_importances_RF = pd.DataFrame(rf_classifier.feature_importances_, index=X.columns, columns=['Importance'])

print("Feature Importances:")
print(feature_importances_RF)

print("Report of XGBoost")
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


feature_importances_XG = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance'])
print("Feature Importances:")
print(feature_importances_XG)

df_final_dengue_data = dengue_data_final[["Month of affection", "City of affection", "Gender", "Age_Group", "Temperature_Group", "Headache_EyePain_Fatigue_MCA", 
                                         "Muscle_Joint_Belly_Pain_MCA", "Appetite_Loss/Metalic_Taste_MCA", "Nausea/vomiting_Diarrhea_MCA", 
                                         "Rash", "Blood in vomit or stool", "Bleeding gums or nose", "Swollen glands", "Dengue Fever"]]


month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_wise_weight_list = {}
index = 0
for i in month_list:
    month_wise_weight_list[i] = prediction_list[index]
    index+= 1

print(month_wise_weight_list)

for i in range(len(df_final_dengue_data["Month of affection"])):
    month_value = month_wise_weight_list[df_final_dengue_data.at[i, "Month of affection"]]
    df_final_dengue_data.at[i, "Month of affection"] = month_value
    city_value = month_value * district_wise_weight[df_final_dengue_data.at[i, "City of affection"]]
    df_final_dengue_data.at[i, "City of affection"] = city_value

imputer = SimpleImputer(strategy="most_frequent")
df_final_dengue_data = pd.DataFrame(imputer.fit_transform(df_final_dengue_data), columns=df_final_dengue_data.columns)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


label_encoders = {}
categorical_cols = ["Gender", "Age_Group", "Temperature_Group", "Rash", "Blood in vomit or stool", "Bleeding gums or nose", "Swollen glands", "Dengue Fever"]
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df_final_dengue_data[col] = label_encoders[col].fit_transform(df_final_dengue_data[col])


numeric_cols = ["Headache_EyePain_Fatigue_MCA", "Muscle_Joint_Belly_Pain_MCA", "Appetite_Loss/Metalic_Taste_MCA", "Nausea/vomiting_Diarrhea_MCA", "Month of affection", "City of affection"]
df_final_dengue_data[numeric_cols] = df_final_dengue_data[numeric_cols].astype(float)

X = df_final_dengue_data.drop("Dengue Fever", axis=1)
y = df_final_dengue_data["Dengue Fever"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
print("Report of Random Forest")
print(classification_report(y_test, y_pred))

feature_importances_RF = pd.DataFrame(rf_classifier.feature_importances_, index=X.columns, columns=['Importance'])

print("Feature Importances:")
print(feature_importances_RF)

print("Report of XGBoost")
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


feature_importances_XG = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance'])
print("Feature Importances:")
print(feature_importances_XG)