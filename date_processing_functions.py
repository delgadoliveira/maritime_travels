import re
import datetime
import numpy as np
import pandas as pd
import seaborn as sns

def _convert_date_two_days(str_date):
    regex_str = '^[0-9]+\/[0-9]+-[0-9]+\/[0-9]+$'
    
    # check if date is in format 4/1-5/13
    if re.findall(regex_str, str_date):
        str_date = str_date.split('/')
        month = str_date[0]
        year = str_date[2]
        day = str_date[1].split('-')[0]
        try:
            new_date = pd.to_datetime(f"{month}/{day}/{year}",format='%m/%d/%y')
            return new_date
        except:
            return np.datetime64('NAT')


def process_book_date(x, load=False, destination=False):
    if load:
        column = 'Load Booking Date'
    if destination:
        column = 'Destination Booking Date'
    
    # if date is already expected format return date
    if (isinstance(x[column], datetime.date)) and not(pd.isnull(x[column])):
        return x[column]

    x[column] = str(x[column])
    return _convert_date_two_days(x[column])


def get_convertible_rows(df):
    for column in ['Load Current', 'Destination Current']:
        df[column] = df[column].astype(str)
        df['Processed '+column] = 'NaN'
        df['Processed '+column] = df.loc[
            (
                df[column].str.contains('[0-9]',regex=True, na=False) |
                df[column].str.contains('[0-9]{4}-[0-9]{2}-[0-9]{2}',regex=True, na=False)
            ), column]
    return df


def get_date(x, load=False, destination=False):
    if load:
        booking_column = 'Load Booking Date'
        realized_column = 'Processed Load Current'
    if destination:
        booking_column = 'Destination Booking Date'
        realized_column = 'Processed Destination Current'
    str_date = str(x[realized_column]).lower()
    #case 16/04/2019  00:00:00
    try:
        str_date = pd.to_datetime(x[realized_column])
        return str_date
    except:
        pass    
    
    if isinstance(x[booking_column], datetime.date):
        if any(k in str_date for k in ['loaded','completed','arrived']):
            str_date = str_date.split(' ')
            if len(str_date)>1:
                try:
                    #case loaded 2/1
                    str_date = pd.to_datetime(f"{str_date[1]}/{x[booking_column].year}"
                                      ,format='%m/%d/%Y')
                    if (x[booking_column].month == 12) and (str_date.day < x[booking_column].day):
                        str_date += pd.DateOffset(years=1)
                    return str_date
                except:
                    pass 
                try:
                    #case loaded 7-27
                    str_date = pd.to_datetime(f"{str_date[1]}-{x[booking_column].year}"
                                      ,format='%m-%d-%Y')
                    return str_date
                except:
                    pass    
                try:
                    #case loaded 27 June
                    str_date = pd.to_datetime(f"{str_date[2]}/{str_date[1]}/{x[booking_column].year}"
                                       ,format='%B/%d/%Y')
                    return str_date
                except:
                    pass
                try:
                    #case loaded 2 Aug
                    str_date = pd.to_datetime(f"{str_date[2]}/{str_date[1]}/{x[booking_column].year}"
                                       ,format='%b/%d/%Y')
                    return str_date
                except:
                    pass
                return np.datetime64('NAT')
     
    return _convert_date_two_days(x[realized_column])   

def split_dates(df):
    date_df= df.select_dtypes(include='datetime64[ns]')
    for col in date_df.columns:
        df[f'{col}_day'] = date_df[col].dt.day
        df[f'{col}_month'] = date_df[col].dt.month
        df[f'{col}_year'] = date_df[col].dt.year
        df[f'{col}_quarter'] = date_df[col].dt.quarter
        df[f'{col}_dayofweek'] = date_df[col].dt.dayofweek
        df[f'{col}_dayofyear'] = date_df[col].dt.dayofyear
    
    return df.drop(columns=date_df.columns)

def strip_categoricals(df):
    cat_cols = df.select_dtypes(include='object').columns.values
    df[cat_cols] = df[cat_cols].apply(lambda x: x.str.strip(), axis=1)    
    return df

def encode_categoricals(df):
    categorical_features = df.select_dtypes(include='object').columns.values
    one_hot_df = pd.get_dummies(df[categorical_features])
    return pd.merge(
        df.drop(columns=categorical_features),
        one_hot_df,
        left_index=True,
        right_index=True
        )

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    return df[((df[column] > (Q1 - 1.5 * IQR)).all(axis=1) & (df[column] < (Q3 + 1.5 * IQR)).all(axis=1))]

def get_seg_metrics(df, column):
    return df[[column,'expected time','realized time','time error']]\
        .groupby(column)\
            .agg({
                'expected time':['mean','median','std'],
                'realized time':['mean','median','std'],
                'time error':['mean','median','std']
        })

def ret_cross_table_row(x):
    cross_ = np.array([[0,0,0], [0,0,0], [0,0,0]])
    # early_arrive and early_load
    if (x['load_delay']<0) and (x['destination_delay']<0):
        cross_[0][0]=1
    # early_arrive and timely_load
    elif (x['load_delay']==0) and (x['destination_delay']<0):
        cross_[1][0] =1
    # early_load and timely_arrive
    elif (x['load_delay']<0) and (x['destination_delay']==0):
        cross_[0][1] =1
    # timely_load and timely_arrive
    elif (x['load_delay']==0) and (x['destination_delay']==0):
        cross_[1][1] =1
    # late_load and early_arrive
    elif (x['load_delay']>0) and (x['destination_delay']<0):
        cross_[2][0] =1
    # early_load and late_arrive
    elif (x['load_delay']<0) and (x['destination_delay']>0):
        cross_[0][2] =1
    # late_load and timely_arrive
    elif (x['load_delay']>0) and (x['destination_delay']==0):
        cross_[2][1] =1
    # timely_load and late_arrive
    elif (x['load_delay']==0) and (x['destination_delay']>0):
        cross_[1][2] =1
    # late_load and late_arrive
    elif (x['load_delay']>0) and (x['destination_delay']>0):
        cross_[2][2] =1
    return cross_

def ret_cross_table(df):
    ret_df = pd.DataFrame(
        sum(df.apply(lambda x: ret_cross_table_row(x), axis=1)),
        index=['early_load','timely_load','late_load'], 
        columns=['early_arrive','timely_arrive','late_arrive']
    )
    return ret_df 

def line_plot(df, x, y, ax, title=None):
    sns.lineplot(x=x, y=y, data=df, marker="o", ax=ax)
    ax.grid()

    if title:
        ax.set_title(title)

    # add annotations one by one with a loop
    for line in range(0,len(df)):
         ax.text(df[x][line], 
                 df[y][line]+0.02, 
                 round(df[y][line],2),
                 horizontalalignment='left', 
                 size='small', 
                 color='black', 
                 weight='semibold')
    return ax



## Barh plot
# ship_destination = dfTemp[['Ship Nomination','Destination','time error']]\
#     .groupby(['Ship Nomination','Destination'])\
#     .agg({
#         'time error':'mean'
#     }).reset_index()


# fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(20,70))
# plt.subplots_adjust(wspace=1, hspace=None)

# destinations = ship_destination['Destination'].unique()

# for i, d in enumerate(destinations):
#     j = int(i/3)
#     k = i % 3
#     ship_destination[ship_destination['Destination'] == d]\
#         .sort_values(by=['time error'])\
#         .plot.barh(x= 'Ship Nomination', y='time error', ax=axes[j][k])
#     axes[j][k].set_xlabel('Avg Time Error (days)')
#     axes[j][k].grid(axis='x')
#     axes[j][k].set_title(d)
