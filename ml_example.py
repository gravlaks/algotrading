"""
File name: ml_example.py

Creation Date: Tue 03 Aug 2021

Description:

"""

# Python Libraries
# -----------------------------------------------------------------------------

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import tensorflow as tf


from yahoo_fin.stock_info import get_data

# Local Application Modules
# -----------------------------------------------------------------------------

ticker = "aapl"

data = get_data(ticker=ticker, start_date="01/01/2015", end_date="01/01/2020")
data = data.drop(["volume", "adjclose", "ticker"], axis=1)


## relative data change
close_values = data["close"][4::6]
open_values = data["open"][5::6]

x = data.pct_change()[1:]


# We remove date from index and drop the following date column
def every_6_open(row):
    if (row.name+1) % 5 == 0:
        return int(row["open"]>0)
    else:
        return float('NaN')
x.reset_index(inplace=True, drop=True)
x["opened_up_positive"] = x.apply(every_6_open, axis=1)

target = x.opened_up_positive.dropna().values.astype(np.float32)
print("Target", target.shape)

x = x.drop(["opened_up_positive"], axis=1)
#X = x[(x.index+1)%6 == 0]
#print("shape 1,", X.shape[0])

x = x.to_numpy().astype(np.float32)
fivers_div_length = (x.shape[0]//5)*5
x = x[:fivers_div_length]
target = target[:fivers_div_length]
print(x.shape)
x = np.split(x, x.shape[0]//5)
x = np.array(x)


print(x.shape)
print(x.dtype)

x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.2)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x.shape[1], x.shape[2])),
    tf.keras.layers.LSTM(64, input_dim=(x.shape[1], x.shape[2])),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Flatten()
])
model.compile(loss="mean_squared_error", optimizer="Adam", metrics =['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=64, epochs = 500, validation_split=0.2)
predicted = np.where(model.predict(x_test)>0.5, 1, 0)

print(predicted)
print(y_test)


### Evaluation:
print(metrics.classification_report(y_test, predicted))
print("Accuracy", metrics.accuracy_score(y_test, predicted))
print(x[0])



from trade_stat_logger import SimpleLogger

logger = SimpleLogger()

ticker = "aapl"

for i in range(len(predicted)):
    if predicted[i] == 1:
        logger.log(security=ticker, share_price=close_values[i], shares=100)
        logger.log(security=ticker, share_price=open_values[i], shares=-100)
    else:
        logger.log(security=ticker, share_price=open_values[i], shares=+100)
        logger.log(security=ticker, share_price=close_values[i], shares=-100)

logger.graph_statistics()
