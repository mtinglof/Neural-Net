# Stock Neural Network  
> Introduction to stock perdiction using Neural Networks 

This project was an introduction to my use of neural networks and trying to predict stock price changes. The data is gathered using pandas data reader from Yahoo! finance.   

## Installing / Getting started

In addition to using the latest version of Python (this project was developed in Python 3.6.8), several data packages need to be downloaded before the program can run. 

```
pip install numpy
pip install pandas
pip install pandas_datareader.data
pip install scipy
```

### Initial Configuration

The program will initially train itself on twenty-six Fortune Five Hundred companies. It will then call and try and predict next day Tesla stock price value. This perdict company can easily be changed within the build however. 

## Features

* Dynamic build for the network. 
  * Any data frame passed as an input will dynamically change the input size. 
  * The number of nodes within the hidden layer may be changed within the initial parameters of the Train class (the default is set to 25). 
  * Output size is also user set, though output should match passed trained output size.  
* This network does not need to be trained with companies and stock data. The code is completely interchangeable with any passed data. 

## Configuration

An example starting configuration is listed in the code and given here: 

```
companylist = ["AAPL", "BLK", "CF", "DOV", "ETR", "FLT", "GPS", "HOG", "IRM", "JPM", "KIM", "LMT", "M", "NKE", "OXY", "PCAR", "QRVO", 
"RE", "SEE", "TROW", "UNM", "V", "WELL", "XLNX", "YUM", "ZTS"]
companylist = ["TSLA"]
test = Rob(companylist, False)
test.build_set()

train = Train(np.load("test.npy"))
train.gen_theta()
train.train(15)

pred = Rob(["TSLA"], True)
pred.build_set()
train.perdict()
```

## Contributing

Pull requests are gladly accepted. If you would like to contribute to the project, please fork the repository and use a feature branch.

## Links

- Repository: https://github.com/mtinglof/neural-net-cat
- Pandas DataReader: https://pandas-datareader.readthedocs.io/en/latest/
- Yahoo Finance: https://finance.yahoo.com/
- Coursera Machine Learning: https://www.coursera.org/learn/machine-learning
- My Homepage: https://github.com/mtinglof

## Licensing

This project is licensed under Unlicense license. This license does not require you to take the license with you to your project.
