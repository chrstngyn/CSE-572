import pandas as pd
import numpy as np

# Write a Python script that accepts two csv files: CGMData.csv and InsulinData.csv and runs the analysis
# procedure and outputs the metrics discussed in the metrics section in another csv file using the format
# described in Results.csv

data_cgm = pd.read_csv("CGMData.csv", low_memory=False, usecols=["Date", "Time", "Sensor Glucose (mg/dL)"])
data_insulin = pd.read_csv("InsulinData.csv", low_memory=False)

data_cgm["date_time_stamp"]=pd.to_datetime(data_cgm["Date"] + " " + data_cgm["Time"])

data_removed = data_cgm[data_cgm["Sensor Glucose (mg/dL)"].isna()]["Date"].unique()

data_cgm = data_cgm.copy()
data_cgm = data_cgm.set_index("Date").drop(index=data_removed).reset_index()

data_insulin["date_time_stamp"] = pd.to_datetime(data_insulin["Date"] + " " + data_insulin["Time"])

automode_start = (
    data_insulin
    .sort_values(by="date_time_stamp", ascending=True)
    .loc[data_insulin["Alarm"] == "AUTO MODE ACTIVE PLGM OFF"]
    .iloc[0]["date_time_stamp"]
)

get_data_auto = (
    data_cgm
    .sort_values(by='date_time_stamp', ascending=True)
    .loc[data_cgm['date_time_stamp']>=automode_start]
)

get_data_manual = (
    data_cgm
    .sort_values(by="date_time_stamp", ascending=True)
    .loc[data_cgm["date_time_stamp"] < automode_start]
)

# ------------------- AUTO MODE

data_auto = get_data_auto.copy()
data_auto = data_auto.set_index("date_time_stamp")

list_auto = (
    data_auto
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    .where(lambda x: x > 0.8 * 288).dropna().index.tolist()
)

data_auto = data_auto.loc[data_auto["Date"].isin(list_auto)]

# Hyperglycemia (> 180 mg/dL)

auto_hyperglycemia_wholeday_time = (
    data_auto
    .between_time("0:00:00", "23:59:59")[["Date", "Time", "Sensor Glucose (mg/dL)"]]
    .loc[data_auto["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_hyperglycemia_daytime_time = (
    data_auto
    .between_time("6:00:00", "23:59:59")[["Date", "Time", "Sensor Glucose (mg/dL)"]]
    .loc[data_auto["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_hyperglycemia_overnight_time = (
    data_auto
    .between_time("0:00:00", "05:59:59")[["Date", "Time", "Sensor Glucose (mg/dL)"]]
    .loc[data_auto["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

# Hyperglycemia critical (> 250 mg/dL)

auto_critical_hyperglycemia_wholeday_time = (
    data_auto
    .between_time("0:00:00", "23:59:59")[["Date", "Time", "Sensor Glucose (mg/dL)"]]
    .loc[data_auto["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_critical_hyperglycemia_daytime_time = (
    data_auto
    .between_time("6:00:00", "23:59:59")[["Date", "Time", "Sensor Glucose (mg/dL)"]]
    .loc[data_auto["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_critical_hyperglycemia_overnight_time = (
    data_auto
    .between_time("0:00:00", "05:59:59")[["Date", "Time", "Sensor Glucose (mg/dL)"]]
    .loc[data_auto["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

# Range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)

auto_normal_wholeday_time = (
    data_auto
    .between_time("0:00:00", "23:59:59")[["Date", "Time", "Sensor Glucose (mg/dL)"]]
    .loc[(data_auto["Sensor Glucose (mg/dL)"] >= 70) & (data_auto["Sensor Glucose (mg/dL)"] <= 180)]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_normal_daytime_time = (
    data_auto
    .between_time("6:00:00", "23:59:59")[["Date", "Time", "Sensor Glucose (mg/dL)"]]
    .loc[(data_auto["Sensor Glucose (mg/dL)"] >= 70) & (data_auto["Sensor Glucose (mg/dL)"] <= 180)]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_normal_overnight_time = (
    data_auto.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]]
        .loc[
            (data_auto["Sensor Glucose (mg/dL)"] >= 70) 
            & (data_auto["Sensor Glucose (mg/dL)"] <= 180)
            ]
            .groupby("Date")["Sensor Glucose (mg/dL)"]
            .count() / 288 * 100
)

# Range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)

auto_normal_wholeday_time_secondary = (
    data_auto.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (data_auto["Sensor Glucose (mg/dL)"] >= 70)
        & (data_auto["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_normal_daytime_time_secondary = (
    data_auto.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (data_auto["Sensor Glucose (mg/dL)"] >= 70)
        & (data_auto["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_normal_overnight_time_secondary = (
    data_auto.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (data_auto["Sensor Glucose (mg/dL)"] >= 70)
        & (data_auto["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

# Hypoglycemia level 1 (CGM < 70 mg/dL)

auto_hypoglycemia_l1_wholeday_time = (
    data_auto.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_auto["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_hypoglycemia_l1_daytime_time = (
    data_auto.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_auto["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_hypoglycemia_l1_overnight_time = (
    data_auto.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_auto["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

# Hypoglycemia level 2 (CGM < 54 mg/dL)
auto_hypoglycemia_l2_wholeday_time = (
    data_auto.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_auto["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_hypoglycemia_l2_daytime_time = (
    data_auto.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_auto["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

auto_hypoglycemia_l2_overnight_time = (
    data_auto.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_auto["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

# ------------------- MANUAL MODE

data_manual = get_data_manual.copy()
data_manual = data_manual.set_index("date_time_stamp")

list_manual = (
    data_manual.groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    .where(lambda x: x > 0.8 * 288)
    .dropna()
    .index.tolist()
)

# Hyperglycemia (> 180 mg/dL)

manual_hyperglycemia_wholeday_time = (
    data_manual.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_hyperglycemia_daytime_time = (
    data_manual.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_hyperglycemia_overnight_time = (
    data_manual.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

# Hyperglycemia critical (> 250 mg/dL)

manual_critical_hyperglycemia_wholeday_time = (
    data_manual.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_critical_hyperglycemia_daytime_time = (
    data_manual.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_critical_hyperglycemia_overnight_time = (
    data_manual.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)


# Range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)

manual_normal_wholeday_time = (
    data_manual.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (data_manual["Sensor Glucose (mg/dL)"] >= 70)
        & (data_manual["Sensor Glucose (mg/dL)"] <= 180)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_normal_daytime_time = (
    data_manual.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (data_manual["Sensor Glucose (mg/dL)"] >= 70)
        & (data_manual["Sensor Glucose (mg/dL)"] <= 180)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_normal_overnight_time = (
    data_manual.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (data_manual["Sensor Glucose (mg/dL)"] >= 70)
        & (data_manual["Sensor Glucose (mg/dL)"] <= 180)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)


# Range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)

manual_normal_wholeday_time_secondary = (
    data_manual.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (data_manual["Sensor Glucose (mg/dL)"] >= 70)
        & (data_manual["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_normal_daytime_time_secondary = (
    data_manual.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (data_manual["Sensor Glucose (mg/dL)"] >= 70)
        & (data_manual["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)


manual_normal_overnight_time_secondary = (
    data_manual.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (data_manual["Sensor Glucose (mg/dL)"] >= 70)
        & (data_manual["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

# Hypoglycemia level 1 (CGM < 70 mg/dL)
manual_hypoglycemia_l1_wholeday_time = (
    data_manual.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_hypoglycemia_l1_daytime_time = (
    data_manual.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_hypoglycemia_l1_overnight_time = (
    data_manual.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

# Hypoglycemia level 2 (CGM < 54 mg/dL)

manual_hypoglycemia_l2_wholeday_time = (
    data_manual.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_hypoglycemia_l2_daytime_time = (
    data_manual.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

manual_hypoglycemia_l2_overnight_time = (
    data_manual.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[data_manual["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count() / 288 * 100
)

results = pd.DataFrame(
    {
        # Overnight

        "Percent Time: Hyperglycemia Overnight (CGM > 180 mg/dL)":[ 
            manual_hyperglycemia_overnight_time.mean(axis=0),
            auto_hyperglycemia_overnight_time.mean(axis=0)
        ],
        
        "Percent Time: Hyperglycemia Critical Overnight (CGM > 250 mg/dL)":[
            manual_critical_hyperglycemia_overnight_time.mean(axis=0),
            auto_critical_hyperglycemia_overnight_time.mean(axis=0)
        ],


        "Percent Time: Overnight Range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)":[ 
            manual_normal_overnight_time.mean(axis=0),
            auto_normal_overnight_time.mean(axis=0)
        ],
                
        "Percent Time: Overnight Range Secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)":[ 
            manual_normal_overnight_time_secondary.mean(axis=0),
            auto_normal_overnight_time_secondary.mean(axis=0)
        ],

        "Percent Time: Hypoglycemia L1 Overnight (CGM < 70 mg/dL)":[ 
            manual_hypoglycemia_l1_overnight_time.mean(axis=0),
            auto_hypoglycemia_l1_overnight_time.mean(axis=0)
        ],
            
        "Percent Time: Hypoglycemia L2 Overnight (CGM < 54 mg/dL)":[ 
            np.nan_to_num(
                manual_hypoglycemia_l2_overnight_time.mean(axis=0)),
                auto_hypoglycemia_l2_overnight_time.mean(axis=0)
        ],

        # Daytime

        "Percent Time: Hyperglycemia Daytime (CGM > 180 mg/dL)":[ 
            manual_hyperglycemia_daytime_time.mean(axis=0),
            auto_hyperglycemia_daytime_time.mean(axis=0)
        ],

        "Percent Time: Hyperglycemia Critical Daytime (CGM > 250 mg/dL)":[
            manual_critical_hyperglycemia_daytime_time.mean(axis=0),
            auto_critical_hyperglycemia_daytime_time.mean(axis=0)
        ],

        "Percent Time: Daytime Range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)":[ 
            manual_normal_daytime_time.mean(axis=0),
            auto_normal_daytime_time.mean(axis=0)
        ],
                           
        "Percent Time: Daytime Range Secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)":[ 
            manual_normal_daytime_time_secondary.mean(axis=0),
            auto_normal_daytime_time_secondary.mean(axis=0)
        ],
                           
        "Percent Time: Hypoglycemia L1 Daytime (CGM < 70 mg/dL)":[ 
            manual_hypoglycemia_l1_daytime_time.mean(axis=0),
            auto_hypoglycemia_l1_daytime_time.mean(axis=0)
        ],
        
        "Percent Time: Hypoglycemia L2 Daytime (CGM < 54 mg/dL)":[ 
            manual_hypoglycemia_l2_daytime_time.mean(axis=0),
            auto_hypoglycemia_l2_daytime_time.mean(axis=0)
        ],

        # Wholeday
                           
        "Percent Time: Hyperglycemia Wholeday (CGM > 180 mg/dL)":[ 
            manual_hyperglycemia_wholeday_time.mean(axis=0),
            auto_hyperglycemia_wholeday_time.mean(axis=0)
        ],
        
        "Percent Time: Hyperglycemia Critical Wholeday (CGM > 250 mg/dL)":[
            manual_critical_hyperglycemia_wholeday_time.mean(axis=0),auto_critical_hyperglycemia_wholeday_time.mean(axis=0)
        ],
        
        "Percent Time: Wholeday Range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)":[ 
            manual_normal_wholeday_time.mean(axis=0),
            auto_normal_wholeday_time.mean(axis=0)
        ],
        
        "Percent Time: Wholeday Range Secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)":[ 
            manual_normal_wholeday_time_secondary.mean(axis=0),
            auto_normal_wholeday_time_secondary.mean(axis=0)
        ],

        "Percent Time: Hypoglycemia L1 Wholeday (CGM < 70 mg/dL)":[ 
            manual_hypoglycemia_l1_wholeday_time.mean(axis=0),
            auto_hypoglycemia_l1_wholeday_time .mean(axis=0)
        ],
        
        "Percent Time: Hypoglycemia L2 Wholeday (CGM < 54 mg/dL)":[ 
            manual_hypoglycemia_l2_wholeday_time.mean(axis=0),
            auto_hypoglycemia_l2_wholeday_time.mean(axis=0)
        ]
                    
    },
    
    index=["Manual Mode", "Auto Mode"]
)

results.to_csv('Results.csv', index=False, header=False)
