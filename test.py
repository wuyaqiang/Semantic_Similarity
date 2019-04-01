import pandas as pd

rnn_0 = pd.read_csv('./training_data/rnn_result/vel0', header=0)
rnn_1 = pd.read_csv('./training_data/rnn_result/vel1', header=0)
rnn_2 = pd.read_csv('./training_data/rnn_result/vel2', header=0)
rnn_3 = pd.read_csv('./training_data/rnn_result/vel3', header=0)
rnn_4 = pd.read_csv('./training_data/rnn_result/vel4', header=0)
rnn_5 = pd.read_csv('./training_data/rnn_result/vel5', header=0)
rnn_6 = pd.read_csv('./training_data/rnn_result/vel6', header=0)
rnn_7 = pd.read_csv('./training_data/rnn_result/vel7', header=0)
rnn_8 = pd.read_csv('./training_data/rnn_result/vel8', header=0)
rnn_9 = pd.read_csv('./training_data/rnn_result/vel9', header=0)

rnn_model = pd.concat([rnn_0, rnn_1, rnn_2,rnn_3,rnn_4,rnn_5,rnn_6,rnn_7,rnn_8,rnn_9])

rnn_model.to_csv('./training_data/rnn_model.csv', index=False)