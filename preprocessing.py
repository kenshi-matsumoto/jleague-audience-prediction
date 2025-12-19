import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.model_selection import KFold
import datetime
from japanmap import groups
from japanmap import pref_code


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
venue_information = pd.read_csv("ex_venue_information.csv")
match_reports = pd.read_csv("match_reports.csv")
holidays_in_japan = pd.read_csv("holidays_in_japan.csv")
pref_lat_lon = pd.read_csv("pref_lat_lon.csv")
team_pref = pd.read_csv("team_pref.csv")
team_standings = pd.read_csv("team_standings_min.csv")

train = pd.concat([train, test])

venue_information['address'] = venue_information['address'].replace(re.compile(r'(府.*)|(県.*)'), '', regex=True)
venue_information['address'] = venue_information['address'].replace(re.compile(r'東京都.*'), '東京都', regex=True)
venue_information['address'] = venue_information['address'].replace(re.compile(r'北海道.*'), '北海道', regex=True)
pref_lat_lon['pref_name'] = pref_lat_lon['pref_name'].replace(re.compile(r'(府.*)|(県.*)'), '', regex=True)

def grouping(pref):
    grp_list = ['北海道', '東北', '関東', '中部', '近畿', '中国', '四国', '九州']
    for grp in grp_list:
        if pref_code(pref) in groups[grp]:
            out = grp
    return out

venue_information['group'] = venue_information['address'].apply(grouping)

train = train.merge(venue_information, on='venue')

match_reports = match_reports[['id', 'home_team_score', 'away_team_score']]
train = train.merge(match_reports, on='id')

#train['away_team_score'] = train['away_team_score'].replace({8:6, 7:6})
#train['home_team_score'] = train['home_team_score'].replace({8:6, 7:6})

train['match_date'] = pd.to_datetime(train['match_date'])
holidays_in_japan['holiday_date'] = pd.to_datetime(holidays_in_japan['holiday_date'])
train['is_holiday'] = train['match_date'].isin(holidays_in_japan['holiday_date'])
train['week'] = train['match_date'].dt.dayofweek
train['week'] = train['week'].mask(train['is_holiday'], 7)
train = train.drop('is_holiday', axis=1)
train['holiday'] = train['week'].replace({0:'w1', 1:'w1', 2:'w1', 3:'w1', 4:'w1', 5:'sata', 6:'sun', 7:'h'})
train['holiday'] = train['holiday'].replace({'w1':0, 'sata':1, 'sun':2, 'h':3})
train = train.drop('week', axis=1)

#holidays_in_japan = holidays_in_japan.drop('discription', axis=1)
#holidays_in_japan = holidays_in_japan.rename(columns={'holiday_date':'match_date'})
#ptrain = train.merge(holidays_in_japan, on='match_date')

def check_next_day(d):
    next_day = d + datetime.timedelta(days=1)
    if next_day.weekday() >= 5 or next_day in holidays_in_japan['holiday_date']:
        if d.weekday() >= 5 or d in holidays_in_japan['holiday_date']:
            return '土'
        else:
            return '金'
    else:
        if d.weekday() >= 5 or d in holidays_in_japan['holiday_date']:
            return '日'
        else:
            return '平日'

#train['day_type'] = train['match_date'].apply(check_next_day)
#train['day_type'] = train['day_type'].replace({'金':'平日'})

train['holiday'].value_counts()

#sns.boxplot(data=train, x='holiday', y='attendance')
#plt.show()

venue_sort = train.sort_values('capacity')
venue_dict = {}
count = 1

for name in venue_sort['venue']:
    venue_dict[name] = count
    count +=1

train['venue_encoded'] = train['venue'].replace(venue_dict)

train['weather'] = train['weather'].replace(re.compile(r'雨.*'), '雨', regex=True)
train['weather'] = train['weather'].replace(re.compile(r'曇.*'), '曇', regex=True)
train['weather'] = train['weather'].replace(re.compile(r'晴.*'), '晴', regex=True)
train['weather'] = train['weather'].replace(re.compile(r'霧.*'), '曇', regex=True)
train['weather'] = train['weather'].replace(re.compile(r'雷雨.*'), '雨', regex=True)
train['weather'] = train['weather'].replace(re.compile(r'雪.*'), '雨', regex=True)
#print(train['weather'].unique().tolist())

train['weather_encoded'] = train['weather'].replace({'晴':2, '雨':0, '曇':1, '屋内':3})
train = train.drop('weather', axis = 1)

#print(train['kick_off_time'].unique().tolist())

train['kick_off_time'] = train['kick_off_time'].replace(re.compile(r':'), '', regex=True)
#print(train['kick_off_time'].unique().tolist())

train['kick_off_time'] = train['kick_off_time'].astype(int)
#train['kick_off_time'] = train['kick_off_time'].replace({13:1, 14:1, 15:2, 16:2, 17:2, 18:3, 18:3})

#print(train['home_team'].unique().tolist())
train['home_team'] = train['home_team'].replace({'Ｇ大阪':'G大阪', 'Ｃ大阪':'C大阪', '川崎Ｆ':'川崎F'})

#print(train['home_team'].unique().tolist())
#print(pref_lat_lon['pref_name'].unique().tolist())

train['away_team'] = train['away_team'].replace({'Ｇ大阪':'G大阪', 'Ｃ大阪':'C大阪', '川崎Ｆ':'川崎F'})

team_pref = team_pref.merge(pref_lat_lon, on='pref_name')
home_team_pref = team_pref.rename(columns={'team_name':'home_team', 'pref_name':'home_pref_name', 'lat':'home_lat', 'lon':'home_lon'})
train = train.merge(home_team_pref, on='home_team')
away_team_pref = team_pref.rename(columns={'team_name':'away_team', 'pref_name':'away_pref_name', 'lat':'away_lat', 'lon':'away_lon'})
train = train.merge(away_team_pref, on='away_team')

train['broadcasters'] = train['broadcasters'].str.split('/')
train['broadcasters'] = train['broadcasters'].apply(len)

#train['broadcasters'] = train['broadcasters'].replace({8:5, 7:5, 6:5})

train['match_date'] = train['match_date'].astype(str)
match_date_df = train['match_date'].str.split('-', expand=True)

train = train.drop('match_date', axis=1)
train = train.join(match_date_df)
train = train.rename(columns={0: 'match_year', 1:'match_month', 2:'match_day'})

train['match_year'] = train['match_year'].astype(int)
train['match_month'] = train['match_month'].astype(int)
train['match_day'] = train['match_day'].astype(int)

train['match_month_apr'] = train['match_month'] == 4
train['match_month_oct'] = train['match_month'] == 10
train['match_month_dec'] = train['match_month'] == 12
train['match_month_mar'] = train['match_month'] == 3

def convert_range(x):
    if 1 <= x <= 7:
        return 1
    elif 8 <= x <= 14:
        return 2
    elif 15 <= x <= 21:
        return 3
    elif 22 <= x <= 31:
        return 4

train['match_day_encoded'] = train['match_day'].apply(convert_range)

train = pd.merge(train, team_standings, left_on=['id', 'home_team'], right_on=['id', 'team_name'])
train = pd.merge(train, team_standings, left_on=['id', 'away_team'], right_on=['id', 'team_name'])

train = train.drop('team_name_x', axis = 1)
train = train.drop('team_name_y', axis = 1)
train = train.rename(columns={'win_points_x': 'home_win_points', 'win_points_y': 'away_win_points', 'win_streak_x':'home_win_streak', 'win_streak_y':'away_win_streak', 'prev_score_x':'home_prev_score', 'prev_score_y':'away_prev_score', 'standing_x':'home_standing', 'standing_y':'away_standing'})

le = LabelEncoder()
train = train.drop('venue', axis = 1)
train['address_encoded'] = le.fit_transform(train['address'])
train = train.drop('address', axis = 1)
train['group_encoded'] = le.fit_transform(train['group'])
train = train.drop('group', axis = 1)
train['home_pref_name_encoded'] = le.fit_transform(train['home_pref_name'])
train = train.drop('home_pref_name', axis = 1)
train['away_pref_name_encoded'] = le.fit_transform(train['away_pref_name'])
train = train.drop('away_pref_name', axis = 1)

train['section_encoded'] = train['section'].replace({'第1節':1, '第2節':2, '第3節':3, '第4節':4, '第5節':5, '第6節':6, '第7節':7, '第8節':8, '第9節':9, '第10節':10, '第11節':11, '第12節':12, '第13節':13, '第14節':14, '第15節':15, '第16節':16, '第17節':17, '第18節':18, '第19節':19, '第20節':20, \
                                                    '第21節':21,'第22節':22,'第23節':23,'第24節':24,'第25節':25,'第26節':26,'第27節':27,'第28節':28,'第29節':29,'第30節':30,'第31節':31,'第32節':32,'第33節':33,'第34節':34,})

train['round_encoded'] = train['round'].replace({'第1日':1, '第2日':2, '第3日':3, '第4日':4, '第5日':5})

train['home_team_encoded'] = train['home_team'].replace({'G大阪':1, '甲府':2, 'FC東京':3, '東京V':4, '磐田':5, '清水':6, '名古屋':7, '大宮':8, '浦和':9, '川崎F':10, '広島':11, '横浜FM':12, '横浜FC':13, '千葉':14, '新潟':15, '鹿島':16, '京都':17, '福岡':18, '大分':19, 'C大阪':20, '松本':21, '柏':22, '神戸':23, '札幌':24, '山形':25, '湘南':26, '仙台':27, '鳥栖':28, '徳島':0, '長崎':17})
train['away_team_encoded'] = train['away_team'].replace({'G大阪':1, '甲府':2, 'FC東京':3, '東京V':4, '磐田':5, '清水':6, '名古屋':7, '大宮':8, '浦和':9, '川崎F':10, '広島':11, '横浜FM':12, '横浜FC':13, '千葉':14, '新潟':15, '鹿島':16, '京都':17, '福岡':18, '大分':19, 'C大阪':20, '松本':21, '柏':22, '神戸':23, '札幌':24, '山形':25, '湘南':26, '仙台':27, '鳥栖':28, '徳島':0, '長崎':17})

#sns.boxplot(data=train, x='home_team_encoded', y='attendance')
#plt.show()

train['derby'] = 0
train.loc[((train['away_team']=='清水') & (train['home_team']=='磐田')) | ((train['home_team']=='清水') & (train['away_team']=='磐田')), 'derby'] = 1
train.loc[((train['away_team']=='C大阪') & (train['home_team']=='G大阪')) | ((train['home_team']=='C大阪') & (train['away_team']=='G大阪')), 'derby'] = 2
train.loc[((train['away_team']=='FC東京') & (train['home_team']=='川崎F')) | ((train['home_team']=='FC東京') & (train['away_team']=='川崎F')), 'derby'] = 3
train.loc[(train['home_team']=='浦和') | (train['away_team']=='浦和'), 'derby'] = 4

for i in range(0, 4284):
    v = ((train.at[i, 'lat']), (train.at[i, 'lon']))
    h = ((train.at[i, 'home_lat']), (train.at[i, 'home_lon']))
    a = ((train.at[i, 'away_lat']), (train.at[i, 'away_lon']))
    home_dist = geodesic(h, v).m
    away_dist = geodesic(a, v).m
    train.at[0+i, 'home_distance'] = home_dist
    train.at[0+i, 'away_distance'] = away_dist

train['av_home_win_points'] = train['home_win_points'] / train['section_encoded']
train['av_away_win_points'] = train['away_win_points'] / train['section_encoded']

team_list = ['G大阪', '甲府', 'FC東京', '東京V', '磐田', '清水', '名古屋', '大宮', '浦和', '川崎F', '広島', '横浜FM', '横浜FC', '千葉', '新潟', '鹿島', '京都', '福岡', '大分', 'C大阪', '松本', '柏', '神戸', '札幌', '山形', '湘南', '仙台', '鳥栖', '徳島', '長崎']

for year in range(2006, 2020):
    for team_name in team_list:
        home_team_df = train[(train['home_team']==team_name) & (train['match_year']==year)]
        away_team_df = train[(train['away_team']==team_name) & (train['match_year']==year)]

        team_mean = ( home_team_df['home_standing'].mean() + away_team_df['away_standing'].mean() ) / 2
        train.loc[(train['home_team']==team_name) & (train['match_year']==year) & (train['home_standing'].isnull()), 'home_standing'] = team_mean
        train.loc[(train['away_team']==team_name) & (train['match_year']==year) & (train['away_standing'].isnull()), 'away_standing'] = team_mean
#train['away_standing'].isnull().values.sum()

train = train.sort_values('id')
test = train[3672:]
train = train[:3672]

cat_cols = []
#cat_cols = ['day_type']

for c in cat_cols:
    data_tmp = pd.DataFrame({c:train[c], 'target':train['attendance']})
    data_tmp['target'] = np.log1p(data_tmp['target'])
    target_mean = data_tmp.groupby(c)['target'].mean()
    test[c] = test[c].map(target_mean)

    tmp = np.repeat(np.nan, train.shape[0])

    kf = KFold(n_splits=5, shuffle=True)
    for idx_1, idx_2 in kf.split(train):
        terget_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        tmp[idx_2] = train[c].iloc[idx_2].map(target_mean)

    train[c] = tmp

train = pd.concat([train, test])

train = train.drop('section', axis=1)
train = train.drop('round', axis=1)
train = train.drop('home_team', axis=1)
train = train.drop('away_team', axis=1)

corr = train.corr()
#plt.figure(figsize=(10, 8))  # ヒートマップのサイズを指定
#sns.heatmap(corr, annot=False, cmap='coolwarm')  # ヒートマップを描画
#plt.show()

data_file = "data_preprocessed.csv"
train.to_csv(data_file, index=False)

train = train.drop('venue_encoded', axis=1)
#train = train.drop('weather_encoded', axis=1)
#train = train.drop('section_encoded', axis=1)
train = train.drop('round_encoded', axis=1)
#train = train.drop('kick_off_time', axis=1)
#train = train.drop('holiday', axis=1)
train = train.drop('temperature', axis=1)
#train = train.drop('match_year', axis=1)
#train = train.drop('match_month', axis=1)
#train = train.drop('match_day', axis=1)
train = train.drop('match_day_encoded', axis=1)
#train = train.drop('humidity', axis=1)
#train = train.drop('broadcasters', axis=1)
train = train.drop('home_team_score', axis=1)
train = train.drop('away_team_score', axis=1)
#train = train.drop('home_team_encoded', axis=1)
#train = train.drop('away_team_encoded', axis=1)
train = train.drop('home_pref_name_encoded', axis=1)
train = train.drop('away_pref_name_encoded', axis=1)
#train = train.drop('address_encoded', axis=1)
#train = train.drop('group_encoded', axis=1)
#train = train.drop('lat', axis=1)
#train = train.drop('lon', axis=1)
train = train.drop('home_lat', axis=1)
train = train.drop('home_lon', axis=1)
train = train.drop('away_lat', axis=1)
train = train.drop('away_lon', axis=1)
train = train.drop('home_distance', axis=1)
#train = train.drop('away_distance', axis=1)

train = train.drop('home_win_points', axis=1)
train = train.drop('home_win_streak', axis=1)
train = train.drop('home_prev_score', axis=1)
train = train.drop('home_standing', axis=1)
train = train.drop('av_home_win_points', axis=1)

train = train.drop('away_win_points', axis=1)
train = train.drop('away_win_streak', axis=1)
train = train.drop('away_prev_score', axis=1)
train = train.drop('away_standing', axis=1)
train = train.drop('av_away_win_points', axis=1)

train = train.drop('derby', axis=1)
train = train.drop('match_month_apr', axis=1)
train = train.drop('match_month_oct', axis=1)
train = train.drop('match_month_dec', axis=1)
train = train.drop('match_month_mar', axis=1)


train = train.sort_values('id')

test = train[3672:]
train = train[:3672]

test = test.drop('attendance', axis=1)

train = train[train['attendance'] > 2104]

train_file = "train_preprocessed.csv"
train.to_csv(train_file, index=False)
test_file = "test_preprocessed.csv"
test.to_csv(test_file, index=False)
