import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import math
import datetime
from scipy.fftpack import fft
# import matplotlib.pyplot as plt

def get_df():
    insulin_data = pd.read_csv('InsulinData.csv', parse_dates=[['Date','Time']], keep_date_col=True, low_memory=False)
    insulin_df = insulin_data[['Date_Time', 'Index', 'BWZ Carb Input (grams)']]
    insulin_df.loc[:, 'Index']

    glucose_data = pd.read_csv('CGMData.csv', parse_dates=[['Date','Time']], keep_date_col=True, low_memory=False)
    glucose_df = glucose_data[['Date_Time', 'Sensor Glucose (mg/dL)']]

    return insulin_df, glucose_df

def calc_features(df):
    features = pd.DataFrame()

    for i in range(0, df.shape[0]):
        x = df.iloc[i, :].tolist()
        features = features.append({
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
    return features

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

def choose_bin(x, min_carb, total_bins):
    partition = float((x - min_carb)/20)
    bin =  math.floor(partition)
    if bin == total_bins:
        bin = bin - 1
    return bin

def extract_ground_truth(insulin_df, glucose_df):
    meals = []
    meals_df = pd.DataFrame()
    meal_matrix = pd.DataFrame()
    two_hours = 60 * 60 * 2
    thirty_min = 30 * 60
    sensor_time_interval = 30

    bin_matrix = []
    bins = []
    min_carb = 0
    max_carb = 0
    total_bins = 0

    processed_insulin_df = insulin_df.copy()
    processed_glucose_df = glucose_df.copy()

    # process insulin data
    valid_carb_input = processed_insulin_df['BWZ Carb Input (grams)'].notna() & processed_insulin_df['BWZ Carb Input (grams)'] != 0.0
    processed_insulin_df = processed_insulin_df.loc[valid_carb_input][['Date_Time', 'BWZ Carb Input (grams)']]
    processed_insulin_df.set_index(['Date_Time'], inplace = True)
    processed_insulin_df = processed_insulin_df.sort_index().reset_index()

    valid_glucose = processed_glucose_df['Sensor Glucose (mg/dL)'].notna()
    processed_glucose_df = processed_glucose_df.loc[valid_glucose][['Date_Time', 'Sensor Glucose (mg/dL)']]
    processed_glucose_df.set_index(['Date_Time'], inplace = True)
    processed_glucose_df = processed_glucose_df.sort_index().reset_index()

    min_carb = processed_insulin_df['BWZ Carb Input (grams)'].min()
    max_carb = processed_insulin_df['BWZ Carb Input (grams)'].max()
    total_bins = math.ceil((max_carb - min_carb) / 20)

    for i in range(len(processed_insulin_df)):
        carb_input = processed_insulin_df['BWZ Carb Input (grams)'][i]
        selected_bin = choose_bin(carb_input, min_carb, total_bins)
        bins.append(selected_bin)
    
    processed_insulin_df['bin'] = bins

    for i in range(0, len(processed_insulin_df)-1):
        time_diff_seconds = (processed_insulin_df.iloc[i + 1]['Date_Time'] - processed_insulin_df.iloc[i]['Date_Time']).total_seconds()
        if(time_diff_seconds > two_hours):
            meals.append(True)
        else:
            meals.append(False)
        
    meals.append(True)
    meals_df = processed_insulin_df[meals]
    
    for i in range(len(meals_df)):
        lower_bound = meals_df.iloc[i]['Date_Time'] - datetime.timedelta(seconds=thirty_min)
        upper_bound = meals_df.iloc[i]['Date_Time'] + datetime.timedelta(seconds=two_hours)
        is_within_bounds = (processed_glucose_df['Date_Time'] >= lower_bound) & (processed_glucose_df['Date_Time'] < upper_bound)
        bin = meals_df.iloc[i]['bin']
        filtered_glucose_df = processed_glucose_df[is_within_bounds]
        
        if len(filtered_glucose_df.index) == sensor_time_interval:
            filtered_glucose_df = filtered_glucose_df.T
            filtered_glucose_df.drop('Date_Time', inplace=True)
            
            filtered_glucose_df.reset_index(drop=True, inplace=True)
            filtered_glucose_df.columns = list(range(1, 31))
            
            meal_matrix = meal_matrix.append(filtered_glucose_df, ignore_index=True)
            bin_matrix.append(bin)

    meal_matrix = meal_matrix.apply(pd.to_numeric)
    bin_matrix = np.array(bin_matrix)

    return meal_matrix, bin_matrix, total_bins

def build_ground_truth_cluster_matrix(k, clusters, ground_truth):
    cluster_matrix = np.zeros((k, k))
    for i, j in enumerate(ground_truth):
        row = clusters[i]
        col = j
        cluster_matrix[row][col] += 1
    return cluster_matrix

def calc_dbscan_sse(labels, feature_matrix):
    sum = 0
    cluster_size = max(labels)
    for i in range(cluster_size + 1):
        x = feature_matrix[labels == i] - feature_matrix[labels == i].mean(axis=0) 
        sum = np.sum(x ** 2)
    return sum

def calc_cluster_entropy(gtm):
    gtm_sum = gtm.sum()
    bins = gtm.shape[0]
    cluster_entropy = 0
    cluster_sum = 0
    cluster_entropies = []

    for i in range(bins):
        cluster_sum = np.sum(gtm[i])
        if cluster_sum == 0:
            continue
        for j in range(bins):
            if gtm[i,j] == 0:
                continue
            col_sum = gtm[i,j] / cluster_sum
            entropy = -1 * col_sum * np.log2(col_sum)
            cluster_entropy = cluster_entropy + entropy
        cluster_entropies.append((cluster_sum / gtm_sum) * cluster_entropy)
    return np.sum(cluster_entropies)

def calc_cluster_purity(gtm):
    gtm_sum = gtm.sum()
    bins = gtm.shape[0]
    cluster_sum = 0
    cluster_purity = 0
    cluster_max = 0
    cluster_purities = []

    for i in range(bins):
        cluster_max = np.max(gtm[i])
        cluster_sum = np.sum(gtm[i])
        if cluster_sum == 0:
            continue
        cluster_purity = cluster_max / cluster_sum
        cluster_purities.append((cluster_sum / gtm_sum) * cluster_purity)
    return np.sum(cluster_purities)
    
def main(): 
    insulin_df, glucose_df = get_df()

    meal_matrix, bin_matrix, total_bins = extract_ground_truth(insulin_df, glucose_df)

    feature_matrix = calc_features(meal_matrix).to_numpy()
    print('feature matrix: ', feature_matrix)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    print('scaled_features: ', scaled_features)
    
    # calculate k means
    kmeans = KMeans(n_clusters=total_bins, random_state=0).fit(feature_matrix)
    kmeans_centroid_locations = kmeans.cluster_centers_
    kmeans_labels = kmeans.labels_
    kmeans_gtm = build_ground_truth_cluster_matrix(int(total_bins), kmeans_labels, bin_matrix)
    kmeans_sse = kmeans.inertia_
    kmeans_entropy = calc_cluster_entropy(kmeans_gtm)
    kmeans_purity = calc_cluster_purity(kmeans_gtm)

    # using nearest neighbors to determine eps
    # nbrs = NearestNeighbors(n_neighbors=2).fit(feature_matrix)
    # distances, indices = nbrs.kneighbors(feature_matrix)
    # distances = np.sort(distances, axis=0)
    # distances = distances[:,1]
    # plt.plot(distances)
    # plt.show()


    # calculate dbscan 
    default_epsilon = 50 # retrieved by observing plotted euclidian distances
    dbscan = DBSCAN(eps=default_epsilon, min_samples=total_bins, metric="euclidean").fit(feature_matrix)
    dbscan_labels = dbscan.labels_
    dbscan_clusters = len(np.unqiue(dbscan_labels))
    dbscan_outliers = np.sum(np.array(dbscan_labels) == -1, axis=0)
    dbscan_gtm = build_ground_truth_cluster_matrix(int(total_bins), dbscan_labels, bin_matrix)

    # dbscan_labels = change_minus_ones(dbscan_labels_P1, mealData_ext_P1_NP, get_dbscan_means(dbscan_labels_P1, mealData_ext_P1_NP))
    # dbscan_labels = divide_dbscan_labels(dbscan_labels_P1, nP1, mealData_ext_P1_NP)
    # dbscan_labels = combine_dbscan_labels(dbscan_labels_P1, nP1, mealData_ext_P1_NP)

    

    dbscan_sse = calc_dbscan_sse(dbscan_labels, scaled_features)
    
    dbscan_entropy = calc_cluster_entropy(dbscan_gtm)
    dbscan_purity = calc_cluster_purity(dbscan_gtm)


    output = pd.DataFrame(
        [
            [
                kmeans_sse,
                dbscan_sse,
                kmeans_entropy,
                dbscan_entropy,
                kmeans_purity,
                dbscan_purity,
            ]
        ],
        columns=[
            "SSE for KMeans",
            "SSE for DBSCAN",
            "Entropy for KMeans",
            "Entropy for DBSCAN",
            "Purity for KMeans",
            "Purity for DBSCAN",
        ],
    )
    output = output.fillna(0)
    output.to_csv("Results.csv", index=False, header=None)

if __name__ == '__main__':
    main()
