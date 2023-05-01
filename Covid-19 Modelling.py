import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def PreProcessing():
    data = pd.read_csv('../COVID19_data.csv')

    data['day_num'] = (pd.to_datetime(data['Date']) - datetime.datetime(2021, 3, 16)).dt.days  # Day Number from 16th March 2021

    ndata = data[['Date', 'Confirmed', 'Tested', 'First Dose Administered', 'day_num']]

    # Test is the average number of tests done during the past 7 days (i.e., during t − 7 to t − 1)
    ndata.insert(2, 'Test', (ndata.Tested.rolling(8).sum() - ndata.Tested) / 7)

    ndata.insert(2, 'del_confirmed', ndata['Confirmed'].diff())   # .diff() to remove cumulative to per day data
    ndata.insert(2, 'c_bar', (ndata.del_confirmed.rolling(8).sum() - ndata.del_confirmed) / 7)

    ndata.insert(2, 'dV', ndata['First Dose Administered'].diff())

    temp = ndata[ndata['day_num'] >= 0]   # data from 16 March 2021 to 26 April 2021
    df = temp[temp['day_num'] < 42]
    return df[['day_num', 'dV', 'c_bar', 'Test']]


def Constraints(prmts):
    N = 70000000
    if prmts[0] < 0:     # Restricting Beta from being negative
        prmts[0] = 0

    if 0.156 * N >= prmts[4]:  # R(0) constraint
        prmts[4] = 0.156 * N
    elif 0.36 * N <= prmts[4]:
        prmts[4] = 0.36 * N

    if 12 >= prmts[5]:     # CIR(0)) constraint
        prmts[5] = 12
    elif prmts[5] >= 30:
        prmts[5] = 30

    return prmts


def run_avg(arr):  # running seven-day-average
    avg = []

    for i in range(7):   # for 1st 7 days
        x = np.mean(arr[:i + 1])
        if x < 0:
            x = 1
        avg.append(x)

    for i in range(7, 42):   # after take avg of last 7 days
        x = np.mean(arr[i - 7:i])
        if x < 0:
            x = 1
        avg.append(x)

    return np.array(avg)


def SEIRV(init_val):   # return parameters after updation for 42 days

    beta = init_val[0]
    S = [init_val[1]]
    E = [init_val[2]]
    I = [init_val[3]]
    R = [init_val[4]]
# given
    alpha = 1 / 5.8
    gamma = 1 / 5
    epsilon = 0.66
    N = 70000000

    for t in range(0, 41):

        dW = 0   # conditions for ∆W(t)
        if t <= 30:
            dW = R[0] / 30

        dS = -beta * S[t] * I[t] / N - epsilon * dV[t] + dW
        dE = beta * S[t] * I[t] / N - alpha * E[t]
        dI = alpha * E[t] - gamma * I[t]
        dR = gamma * I[t] + epsilon * dV[t] - dW

        if S[t] + dS < 0:      # when S(t) is negative, scale the remaining values so that the total sum is 70000000
            S.append(0)
            scale = N / (E[t] + dE + I[t] + dI + R[t] + dR)
            E.append((E[t] + dE) * scale)
            I.append((I[t] + dI) * scale)
            R.append((R[t] + dR) * scale)
        else:
            S.append(S[t] + dS)
            E.append(E[t] + dE)
            I.append(I[t] + dI)
            R.append(R[t] + dR)

    return [S, E, I, R]


def grad(prmts):
    [beta, S0, E0, I0, R0, CIR0] = prmts
    grad = []

    # perturb beta on either side by ±0.01,
    # perturb CIR(0) on either side by ±0.01,
    # perturb other parameters by ±1
    grad.append((loss_func([beta + 0.01, S0, E0, I0, R0, CIR0]) - loss_func(
        [beta - 0.01, S0, E0, I0, R0, CIR0])) / 0.02)
    grad.append((loss_func([beta, S0 + 1, E0, I0, R0, CIR0]) - loss_func([beta, S0 - 1, E0, I0, R0, CIR0])) / 2)
    grad.append((loss_func([beta, S0, E0 + 1, I0, R0, CIR0]) - loss_func([beta, S0, E0 - 1, I0, R0, CIR0])) / 2)
    grad.append((loss_func([beta, S0, E0, I0 + 1, R0, CIR0]) - loss_func([beta, S0, E0, I0 - 1, R0, CIR0])) / 2)
    grad.append((loss_func([beta, S0, E0, I0, R0 + 1, CIR0]) - loss_func([beta, S0, E0, I0, R0 - 1, CIR0])) / 2)
    grad.append(
        (loss_func([beta, S0, E0, I0, R0, CIR0 + 0.1]) - loss_func([beta, S0, E0, I0, R0, CIR0 - 0.1])) / 0.2)

    return np.array(grad)


def grad_desnt(prmts):
    delta = 0.01
    j = 0

    tdata = PreProcessing()
    # c_bar, Test, dV are made global to use them in future
    global c_bar
    global Test
    global dV
    c_bar = tdata['c_bar'].to_numpy()
    Test = tdata['Test'].to_numpy()
    dV = tdata['dV'].to_numpy()

    loss = loss_func(prmts)
    while (loss > delta):
        prmts = Constraints(prmts - grad(prmts) / (j + 1))   # To keep parameter within constraints
        loss = loss_func(prmts)
        j += 1

    print('Gradient Descent total iterations = ', j, '\nLoss = ', loss)
    return prmts


def loss_func(prmts):

    alpha = 1 / 5.8
    CIR = prmts[5] * Test[0] * np.reciprocal(Test)    # CIR(t)
    E = SEIRV(prmts)[1]
    avg = alpha * np.divide(E, CIR)      # α∆e(t)

    avg = run_avg(avg)

    loss = np.log(c_bar) - np.log(avg)
    sq_error = np.square(loss).sum()
    return sq_error / 42


def New_Cases_Reported():
    data = pd.read_csv('../COVID19_data.csv')

    data['day_num'] = (pd.to_datetime(data['Date']) - datetime.datetime(2021, 3, 16)).dt.days   # Day number from 16th march

    ndata = data[['Confirmed', 'day_num']]

    ndata.insert(2, 'del_confirmed', ndata['Confirmed'].diff())

    df = ndata[ndata['day_num'] >= 0]   # Data from 16th March onwards

    return df['del_confirmed'].to_list()


def Open_Loop(prmts):
    alpha = 1 / 5.8
    gamma = 1 / 5
    epsilon = 0.66
    N = 70000000
    [beta, S0, E0, I0, R0, CIR0] = prmts

    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]

    dV = PreProcessing()['dV'].to_list()  # For remaining values, took delV = 200000
    for t in range(42, 290):
        dV.append(200000)

    for t in range(290):
        # ∆W(t) Conditions
        dW = 0
        if t <= 30:
            dW = R[0] / 30
        if t >= 180:
            dW = (R[t - 179] - R[t - 180]) + epsilon * dV[t]

        dS = -beta * S[t] * I[t] / N - epsilon * dV[t] + dW
        dE = beta * S[t] * I[t] / N - alpha * E[t]
        dI = alpha * E[t] - gamma * I[t]
        dR = gamma * I[t] + epsilon * dV[t] - dW

        if S[t] + dS < 0:
            S.append(0)
            scale = N / (E[t] + dE + I[t] + dI + R[t] + dR)
            E.append((E[t] + dE) * scale)
            I.append((I[t] + dI) * scale)
            R.append((R[t] + dR) * scale)
        elif R[t] + dR < 0:
            R.append(0)
            scale = N / (E[t] + dE + I[t] + dI + S[t] + dS)
            E.append((E[t] + dE) * scale)
            I.append((I[t] + dI) * scale)
            S.append((S[t] + dS) * scale)
        else:
            S.append(S[t] + dS)
            E.append(E[t] + dE)
            I.append(I[t] + dI)
            R.append(R[t] + dR)

    return [S, E, I, R]


def Closed_Loop(prmts):
    [beta1, S0, E0, I0, R0, CIR0] = prmts

    dV = PreProcessing()['dV'].to_list()    # For remaining values, took delV = 200000
    for t in range(42, 290):
        dV.append(200000)

    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]

    alpha = 1 / 5.8
    gamma = 1 / 5
    epsilon = 0.66
    N = 70000000

    for t in range(290):
        beta = beta1
        if t % 7 == 0 and t >= 42:
            if sum(I[t - 7:t]) / 7 < 10000:
                beta = beta1
            elif sum(I[t - 7:t]) / 7 < 25000:
                beta = 2 * beta1 / 3
            elif sum(I[t - 7:t]) / 7 < 100000:
                beta = beta1 / 2
            else:
                beta = beta1 / 3

        # ∆W(t) Conditions
        dW = 0
        if t <= 30:
            dW = R[0] / 30
        if t >= 180:
            dW = (R[t - 179] - R[t - 180]) + epsilon * dV[t - 180]

        dS = -beta * S[t] * I[t] / N - epsilon * dV[t] + dW
        dE = beta * S[t] * I[t] / N - alpha * E[t]
        dI = alpha * E[t] - gamma * I[t]
        dR = gamma * I[t] + epsilon * dV[t] - dW

        if S[t] + dS < 0:
            S.append(0)
            scale = N / (E[t] + dE + I[t] + dI + R[t] + dR)
            E.append((E[t] + dE) * scale)
            I.append((I[t] + dI) * scale)
            R.append((R[t] + dR) * scale)

        elif R[t] + dR < 0:
            R.append(0)
            scale = N / (E[t] + dE + I[t] + dI + S[t] + dS)
            E.append((E[t] + dE) * scale)
            I.append((I[t] + dI) * scale)
            S.append((S[t] + dS) * scale)
        else:
            S.append(S[t] + dS)
            E.append(E[t] + dE)
            I.append(I[t] + dI)
            R.append(R[t] + dR)

    return [S, E, I, R]

def Daily_Cases_plot(res1, res2, res3, res4, res5, CIR0):

    CIR = CIR0 * Test[0] * np.reciprocal(Test)
    CIR = sum(CIR) / 42

    E1 = res1[1]      # E predictions till 31 December 2021
    E2 = res2[1]
    E3 = res3[1]
    E4 = res4[1]
    E5 = res5[1]

    alpha = 1 / 5.8
    predict1 = alpha * np.array(E1) / CIR
    predict2 = alpha * np.array(E2) / CIR
    predict3 = alpha * np.array(E3) / CIR
    predict4 = alpha * np.array(E4) / CIR
    predict5 = alpha * np.array(E5) / CIR
    d = range(291)

    # extrapolated data till 31st sept
    ground_truth = New_Cases_Reported()
    for i in range(188, 291):
        ground_truth.append(1000)   # Took this value quite low as no.of daily cases reduced after 20th september 2021

    plt.figure(figsize=(12, 8))

    plt.plot(d, predict1)
    plt.plot(d, predict2)
    plt.plot(d, predict3)
    plt.plot(d, predict4)
    plt.plot(d, predict5)
    plt.plot(d, ground_truth)

    legend_name = ['open loop β', 'open loop 2β/3', 'open loop β/2', 'open loop β/3', 'closed loop control',
                   'reported cases (ground truth)']
    plt.legend(legend_name, loc="upper right", fontsize=13)
    plt.xlabel('Day Number', fontsize=12)
    plt.ylabel('Daily Cases', fontsize=18)
    plt.show()


def Susceptible_fraction_plot(res1, res2, res3, res4, res5, CIR0):

    S1 = res1[0]   # E predictions till 31 December 2021
    S2 = res2[0]
    S3 = res3[0]
    S4 = res4[0]
    S5 = res5[0]

    N = 70000000
    d = range(291)
    plt.figure(figsize=(12, 8))
    plt.xlim([0, 400])

    plt.plot(d, np.array(S1) / N)
    plt.plot(d, np.array(S2) / N)
    plt.plot(d, np.array(S3) / N)
    plt.plot(d, np.array(S4) / N)
    plt.plot(d, np.array(S5) / N)

    legend_name = ['open loop β', 'open loop 2β/3', 'open loop β/2', 'open loop β/3', 'closed loop control']
    plt.legend(legend_name, loc="upper right", fontsize=13)
    plt.xlabel('Day Number', fontsize=12)
    plt.ylabel('Susceptible people fraction', fontsize=18)
    plt.show()


if __name__ == '__main__':

 # Initial Parameters
    N = 70000000
    beta = 3.41243564864
    e0 = 0.0016
    i0 = 0.0039
    r0 = 0.32
    s0 = 1 - r0 - e0 - i0
    CIR0 = 29.054

    init_val = np.array([beta, s0 * N, e0 * N, i0 * N, r0 * N, CIR0])
    best_prmts = grad_desnt(init_val)
    print('Best Parameters = ', list(best_prmts))

    # Open Loop Control
    prmts = [0.49639073007972884, 47228999.99999971, 97999.9999962104, 272999.99998911645, 22399999.999999844, 29.50884124522309]
    [beta, S0, E0, I0, R0, CIR0] = prmts

    res1 = Open_Loop([beta, S0, E0, I0, R0, CIR0])

    res2 = Open_Loop([(2*beta)/3, S0, E0, I0, R0, CIR0])

    res3 = Open_Loop([(beta)/2, S0, E0, I0, R0, CIR0])

    res4 = Open_Loop([(beta)/3, S0, E0, I0, R0, CIR0])

    # Closed Loop Control
    res5 = Closed_Loop(best_prmts)

    Daily_Cases_plot(res1, res2, res3, res4, res5, CIR0)

    Susceptible_fraction_plot(res1, res2, res3, res4, res5, CIR0)




