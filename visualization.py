import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 시각화 함수 정의
def plot_ts(data, color, alpha, label):
    
    plt.figure(figsize=(11,5))
    plt.plot(data, color=color, alpha=alpha, label=label)
    plt.title("총 인구수 시계열 예측")
    plt.ylabel('총 인구수')
    plt.legend()
    plt.show()

# 시계열 모델(ARIMA)라는 모델을 이용하여 테스트
def ADF_test(data):
	# ADF 실시
    results = adfuller(data, autolag='AIC')
    
    # 통계량
    s = results[0]
    # p-value
    p = results[1]
    # 기각역
    cv = results[4]
    
    # 출력
    # ADF 테스트 결과 및 H0, H1(귀무, 대립가설)을 통해 가설검정을 한다.
    # P-value, Test Statistics라는 수치를 이용한다.
    print('-'*30)
    print('Augemented Dickey-Fuller Test')
    print('H0 : 단위근이 존재한다 (비정상 시계열)')
    print('Ha : 단위근이 없다 (정상 시계열)')
    print('Critical Values : {}'.format(cv))
    print('-'*30)
    print('Test Statistics : {:.4f}'.format(s))
    print('p-value : {:.4f}'.format(p))
    print('-'*30)

def plot_rolling(data, roll_size):
    # 이동평균함수(rolling) - 평균, 표준편차
    roll_mean = data.rolling(window=roll_size).mean()
    roll_std = data.rolling(window=roll_size).std()
    
    # 시각화
    plt.figure(figsize=(11,5))
    plt.plot(data, color='blue', alpha=0.25, label='Original')
    plt.plot(roll_mean, color='black', label='Rolling mean')
    plt.plot(roll_std, color='red', label='Rolling std')
    plt.title('Rolling Mean & Standard Deviation')
    plt.ylabel("Mean Temperature")
    plt.legend()
    plt.show()



def plot_and_error(data, forecast):
    # MSE 계산
    mse = mean_squared_error(data, forecast)
    # 시각화
    plt.figure(figsize=(11,5))
    plt.plot(data, color='blue', alpha=0.25 , label='Original')
    plt.plot(forecast, color='red', label='predicted')
    plt.title("Time Series Forecast")
    plt.ylabel("Total population")
    plt.legend()
    plt.show()

plot_and_error(total_population.loc[:, "서울특별시  (1100000000)"][1:], forecast1)