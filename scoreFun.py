import numpy as np
import pandas
from scipy.integrate import tplquad,dblquad,quad
from scipy.optimize import curve_fit
import scipy.stats as st

if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    filename = 'H:/数据挖掘/16.xlsx'
    data = pandas.read_excel(filename)
    dataset = data.values.tolist()
    dataset = np.array(dataset)
    Pr = []
    for k in range(250):
        X = dataset[np.nonzero(dataset[:, 27] == k)[0]]
        X = X[:, 3:27]
        # print(X)
        n = np.shape(X)[1]
        mu = np.mean(X, axis = 0)
        sigma = np.std(X, axis = 0)
        # print(mu, sigma)

        for i in range(n):
            x = X[:,i]
            Q=[]
            # print(x)
            # Fi = st.norm.cdf(x, mu[i], sigma[i])
            standard = st.norm.cdf(0, mu[i], sigma[i])
            for y in range(len(x)):
                if x[y]> 15:
                    value=1 - (st.norm.cdf(x[y], mu[i], sigma[i])-standard)/(1-standard)
                else:
                    value=1
                Q.append(value)

                test = pandas.DataFrame(data=Q)
                test.to_csv('Pr.csv', mode='w', header=False)