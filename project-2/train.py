import pandas as pd
import numpy as np
from sklearn import decomposition, preprocessing
from sklearn import model_selection
from sklearn import svm
from scipy.fftpack import fft
import pickle
import pickle_compat

pickle_compat.patch()

def read_to_df():
    df_insulin_data = pd.read_csv('InsulinData.csv', low_memory=False)
    df_insulin_patient_data = pd.read_csv('Insulin_patient2.csv',low_memory=False)
    df_cgm_data = pd.read_csv('CGMData.csv',low_memory=False)
    df_cgm_patient_data = pd.read_csv('CGM_patient2.csv',low_memory=False)

    return df_insulin_data, df_insulin_patient_data, df_cgm_data, df_cgm_patient_data

def find_meal_stretch(stretch, time_delta):
    times = []

    stretch_1 = stretch[0:len(stretch)-1]
    stretch_2 = stretch[1:len(stretch)]

    stretch_diff = list(np.array(stretch_1) - np.array(stretch_2))
    
    for i in list(zip(stretch_1, stretch_2, stretch_diff)):
        if i[2] < time_delta:
            times.append(i[0])
    return times

def generate_df(times, start_time, end_time, is_meal, df_glucose):
    meal_list = []

    for time in times:
        meal_index = df_glucose[df_glucose['date_time'].between(time + pd.DateOffset(hours=start_time), time + pd.DateOffset(hours=end_time))]
        if meal_index.shape[0] < 24:
            continue

        glucose_values = meal_index['Sensor Glucose (mg/dL)'].to_numpy()
        mean = meal_index['Sensor Glucose (mg/dL)'].mean()

        if is_meal:
            missing_glucose_value = 30 - len(glucose_values)
            if missing_glucose_value > 0:
                for i in range(missing_glucose_value):
                    glucose_values = np.append(glucose_values, mean)
                meal_list.append(glucose_values[0:30])
        else:
            meal_list.append(glucose_values[0:24])
    return pd.DataFrame(data=meal_list)

def calc_glucose_features(df):
    glucose_features = pd.DataFrame()

    for i in range(0, df.shape[0]):
        x = df.iloc[i, :].tolist()
        glucose_features = glucose_features.append({
            "Minimum Value": min(x),
            "Maximum Value": max(x),
            "Mean of Absolute Values 1": calc_abs_mean(x[:13]),
            "Mean of Absolute Values 2": calc_abs_mean(x[13:]),
            "Root Mean Square": calc_rms(x),
            "Entropy": calc_entropy(x),
            "Max FFT Amplitude 1": calc_fft(x[:13])[0],
            "Max FFT Frequency 1": calc_fft(x[:13])[1],
            "Max FFT Amplitude 2": calc_fft(x[13:])[0],
            "Max FFT Frequency 2": calc_fft(x[13:])[1],
        },
        ignore_index=True
        )
    return glucose_features

def calc_abs_mean(df_row):
    abs_mean = 0
    for i in range(0, len(df_row) - 1):
        abs_mean = abs_mean + np.abs(df_row[(i + 1)] - df_row[i])
    return abs_mean / len(df_row)

def calc_rms(df_row):
    rms = 0
    for i in range(0, len(df_row) - 1):
        rms = rms + np.square(df_row[i])
    return np.sqrt(rms / len(df_row))

def calc_entropy(df_row):
    entropy = 0
    if (len(df_row) <= 1):
        return 0
    else:
        value, index = np.unique(df_row, return_counts=True)
        ratio = index / len(df_row)
        non_zero_ratio = np.count_nonzero(ratio)

        if non_zero_ratio <= 1:
            return 0
        for i in ratio:
            entropy -= i * np.log2(i)
        return entropy

def calc_fft(df_row):
    ffourier = fft(df_row)
    amplitude = []
    frequency = np.linspace(0, len(df_row) * 2/300, len(df_row))

    for amp in ffourier:
        amplitude.append(np.abs(amp))
    
    sorted_amplitude = sorted(amplitude)
    max_amplitude = sorted_amplitude[(-2)]
    max_frequency = frequency.tolist()[amplitude.index(max_amplitude)]
    return [max_amplitude, max_frequency]

def extract_data(df_insulin, df_glucose):
    meal_df = pd.DataFrame()
    meal_times = []
    no_meal_df = pd.DataFrame()
    no_meal_times = []
    standard_scaler = preprocessing.StandardScaler()
    pca = decomposition.PCA(n_components=5)

    df_insulin = df_insulin[::-1]
    df_glucose = df_glucose[::-1]

    df_insulin['date_time'] = pd.to_datetime(df_insulin['Date'] + ' ' + df_insulin['Time'])
    processed_insulin_df = df_insulin[['date_time', 'BWZ Carb Input (grams)']]
    processed_insulin_df = df_insulin[(processed_insulin_df['BWZ Carb Input (grams)'].notna()) & (processed_insulin_df['BWZ Carb Input (grams)'] > 0)]
    time_stretch = list(processed_insulin_df['date_time'])

    df_glucose['Sensor Glucose (mg/dL)'] = df_glucose['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    df_glucose['date_time'] = pd.to_datetime(df_glucose['Date'] + " " + df_glucose['Time'])
    processed_glucose_df = df_glucose[['date_time','Sensor Glucose (mg/dL)']]

    meal_times = find_meal_stretch(time_stretch, pd.Timedelta('0 days 120 min'))
    no_meal_times = find_meal_stretch(time_stretch, pd.Timedelta('0 days 240 min'))

    no_meal_df = generate_df(no_meal_times, 2, 4, False, processed_glucose_df)
    meal_df = generate_df(meal_times, -0.5, 2, True, processed_glucose_df)

    no_meal_features = calc_glucose_features(no_meal_df)
    meal_features = calc_glucose_features(meal_df)


    no_meal_scaler = standard_scaler.fit_transform(no_meal_features)
    meal_scaler = standard_scaler.fit_transform(meal_features)

    no_meal_pca = pd.DataFrame(pca.fit_transform(no_meal_scaler))
    no_meal_pca['class'] = 0

    pca.fit(meal_scaler)
    meal_pca = pd.DataFrame(pca.fit_transform(meal_scaler))
    meal_pca['class'] = 1

    output_data = meal_pca.append(no_meal_pca)
    output_data.index = [i for i in range(output_data.shape[0])]
    return output_data


def main(): 
    insulin_data, insulin_patient_data, cgm_data, cgm_patient_data = read_to_df()

    insulin_data = pd.concat([insulin_patient_data, insulin_data])
    glucose_data = pd.concat([cgm_patient_data, cgm_data])

    df = extract_data(insulin_data, glucose_data)

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)

    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=1)
    for train, test in kfold.split(x, y):
        x_train, x_test = x.iloc[train], x.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        
        model.fit(x_train, y_train)

    with open('model.pkl', 'wb') as (file):
        pickle.dump(model, file)


if __name__ == '__main__':
    main()
