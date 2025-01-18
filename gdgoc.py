# 2025 Gdgoc 백야 해커톤
# import 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import plot_ts, ADF_test

# 시 연령별 인구
city_age_population = pd.read_csv("./dataset/시_연령별_인구.csv")
city_gugo_population = pd.read_csv("./dataset/시군구_연령별_인구.csv")
moving_population = pd.read_csv("./datset/유동인구(수정).csv")
company_population = pd.read_csv("./dataset/직장인구(수수정).csv")
average_cosumption = pd.read_csv("./dataset/평균소득.csv")

# 행정구역, 2019 ~ 2024년에 존재하는 인구수를 열로 잡음.(total_population dataframe)
total_population = city_age_population.loc[:, ["행정구역", "2019년_계_총인구수", "2020년_계_총인구수", "2021년_계_총인구수", "2022년_계_총인구수", "2023년_계_총인구수", "2024년_계_총인구수"]]
total_population = total_population.rename(columns={"2019년_계_총인구수": "2019", "2020년_계_총인구수": "2020", "2021년_계_총인구수":"2021", "2022년_계_총인구수": "2022", "2023년_계_총인구수": "2023", "2024년_계_총인구수":"2024"})
total_population = total_population.set_index("행정구역")
total_population = total_population.transpose()
total_population = total_population.reset_index()
total_population["index"] = total_population["index"].astype(str)
total_population["index"] = pd.to_datetime(total_population["index"])
total_population = total_population.set_index("index")
total_population = total_population.astype(str)
total_population = total_population.astype(np.int64)

# 열인 도시(행정동코드) 형식으로 불러들여서 불러온 다음, Dataframe에 있는 결측값이나 'nan'이라는 문자가 존재.
# 이를 통해 total_population 제거한다.

# 서울에 있는 데이터를 매개로 시각화 그래프(tcf)를 보여준다. => 결과에서도 나오겠지만
# 정상 시계열로 판명되어 매개변수를 변경할 필요가 없음.
plot_ts(total_population.loc[:, "서울특별시  (1100000000)"], 'blue', 0.25, 'Original')
ADF_test(total_population.loc[:, "서울특별시  (1100000000)"])
# ARIMA
from statsmodels.tsa.arima.model import ARIMA

# index를 period로 변환해주어야 warning이 뜨지 않음
ts_copy = total_population.loc[:, "서울특별시  (1100000000)"].copy()
ts_copy.index = pd.DatetimeIndex(ts_copy.index).to_period('D')

# 예측을 시작할 위치(이후 차분을 적용하기 때문에 맞추어주었음
start_idx = ts_copy.index[1]

# ARIMA(1,0,1)
model1 = ARIMA(ts_copy, order=(1,0,1))
# fit model
model1_fit = model1.fit()

# 전체에 대한 예측 실시
forecast1 = model1_fit.predict(start=start_idx)

# 연령별에 대해서 데이터 분별하기
city_age_population_60_69 = city_age_population.loc[:, ['행정구역', '2019년_계_60~69세',	'2020년_계_60~69세',	'2021년_계_60~69세',	'2022년_계_60~69세', '2023년_계_60~69세',	'2024년_계_60~69세']]
city_age_population_60_69 = city_age_population_60_69.rename(columns={"2019년_계_60~69세": "2019", "2020년_계_60~69세": "2020", "2021년_계_60~69세":"2021", "2022년_계_60~69세": "2022", "2023년_계_60~69세": "2023", "2024년_계_60~69세":"2024"})
city_age_population_60_69 = city_age_population_60_69.reset_index()

city_age_population_60_69["index"] = city_age_population_60_69["index"].astype(str)
city_age_population_60_69["index"] = pd.to_datetime(city_age_population_60_69["index"])

city_age_population_60_69 = city_age_population_60_69.set_index("index")
city_age_population_60_69 = city_age_population_60_69.fillna('2000000')

for i in city_list:
  city_age_population_60_69[i] = city_age_population_60_69[i].astype(str).str.replace(',', '')
  city_age_population_60_69[i] = city_age_population_60_69[i].astype(str).str.replace('nan', '2000000')
city_age_population_60_69 = city_age_population_60_69.astype(np.int64)
city_age_population_60_69.head(30)

city_age_population_60_69 = city_age_population_60_69.transpose()
city_age_population_60_69.head(30)

city_age_population_60_69 = city_age_population_60_69.transpose()
city_age_population_60_69.head(30)

# 시각화
plot_ts(city_age_population_60_69.loc[:, "서울특별시  (1100000000)"], 'blue', 0.25, 'Original')

from statsmodels.tsa.stattools import adfuller
adfuller(city_age_population_60_69.loc[:, "서울특별시  (1100000000)"], autolag='AIC')

ADF_test(city_age_population_60_69.loc[:, "서울특별시  (1100000000)"])
# 함수 실행
plot_rolling(city_age_population_60_69.loc[:, "서울특별시  (1100000000)"], 2)

# ARIMA
from statsmodels.tsa.arima.model import ARIMA

# index를 period로 변환해주어야 warning이 뜨지 않음
ts_copy = city_age_population_60_69.loc[:, "서울특별시  (1100000000)"].copy()
ts_copy.index = pd.DatetimeIndex(ts_copy.index).to_period('D')

# 예측을 시작할 위치(이후 차분을 적용하기 때문에 맞추어주었음
start_idx = ts_copy.index[1]

# ARIMA(1,0,1)
model1 = ARIMA(ts_copy, order=(1,0,1))
# fit model
model1_fit = model1.fit()

# 전체에 대한 예측 실시
forecast1 = model1_fit.predict(start=start_idx)
plot_and_error(city_age_population_60_69.loc[:, "서울특별시  (1100000000)"][1:], forecast1)

# 성능 및 평가 분석
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
r2_score(city_age_population_60_69.loc[:, "서울특별시  (1100000000)"][1:].values, forecast1.values)