#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import math


# In[2]:


def Warren_Model(C1, C2, ROP, RPM, d, WOB):
    X1 = np.sqrt(ROP * 12 / (RPM * 60 * d))
    Torque1 = (C1 + C2 * X1) * WOB * d / 12
    return Torque1


# In[3]:


def Pessier_Model(d, WOB, C3, n):
    Torque2 = ((n * d * WOB) / 36) + C3
    return Torque2


# In[4]:


def Patil_Model(cb1, RPM, ksb, kcb, d, WOB):
    n1 = 0.9
    wb = (2 * RPM * np.pi) / 60
    X2 = (ksb - kcb) * np.exp(-n1 * wb)
    Torque3 = (cb1 * wb) + (kcb * wb + X2) * (d / 39.37) * (WOB * 4.448)
    Torque3 = Torque3 / 1.3558  # converting Torque3 from N.m to ft.lbf
    return Torque3


# In[5]:


def Oklahoma_Model(WOB, r, ROP, RPM, k, w, s, e):
    f = ROP * 12 / (60 * RPM)  # converting from ft to in and from hr to min
    Torque4 = ((1 - k * w * s) * e * f * r + k * w * WOB) * r / 24
    return Torque4


# WOB: weight on bit, lbf
# r: bit radius in inches
# k: coefficient of friction
# w: bit constant
# s: ratio of drillstring strength to rock strength
# e: intrinsic specific energy of the rock in psi
# f: depth of cut in in/rpm.
# ROP:rate of penetration, ft/hr
# Torque in ft.lbf


# A Model by Kuwait Oil Company

def Kuwait_Model(WOB, r, ROP, k2, k3, RPM):
    k4 = (2 * RPM * np.pi) / 60
    k1 = (2 * np.pi * ROP) / (3.28 * 3600 * k4)  # converting ROP from ft/hr to m/hr
    Torque5 = (k2 + k3 * np.sqrt(k1 / (r / 39.37))) * (WOB * 4.448) * (
                r / 39.37)  # converting r f/ in to m and WOB f/ lbf to N
    Torque5 = Torque5 / 1.3558  # converting Torque5 from N.m to ft.lbf
    return Torque5


# ROP: rate of pentration, ft/hr
# WOB: weight on bit, lbf
# r: bit radius, in
# k1: depth of cut, m/rad
# k2: bit and formation contact frictional model
# k3: damping coefficient
# k4: bit desired angular speed, rad/s
# Torque in ft.lbf


def calc_torque(model, x, X):
    # model: string containing the name of the model used  for calculating Torque
    # x: array of model parameters
    # X: array of operational parameters
    if model == "warren model":
        C1 = x[0]
        C2 = x[1]

        ROP = X[0]
        RPM = X[1]
        d = X[2]
        WOB = X[3]
        return Warren_Model(C1, C2, ROP, RPM, d, WOB)
    elif model == "pessier model":
        n = x[2]
        C3 = x[3]

        WOB = X[3]
        d = X[2]
        return Pessier_Model(d, WOB, C3, n)
    elif model == "patil model":
        cb1 = x[4]
        ksb = x[5]
        kcb = x[6]

        RPM = X[1]
        d = X[2]
        WOB = X[3]
        return Patil_Model(cb1, RPM, ksb, kcb, d, WOB)
    elif model == "oklahoma model":
        k = x[7]
        w = x[8]
        s = x[9]
        e = x[10]

        ROP = X[0]
        RPM = X[1]
        WOB = X[3]
        r = X[2] / 2
        return Oklahoma_Model(WOB, r, ROP, RPM, k, w, s, e)
    elif model == "kuwait model":
        k2 = x[11]
        k3 = x[12]

        ROP = X[0]
        RPM = X[1]
        WOB = X[3]
        r = X[2] / 2
        return Kuwait_Model(WOB, r, ROP, k2, k3, RPM)
    else:
        raise Unknownmodel("Unknownmodel: Please check the correct syntax or add new model")


# define a general fit fuction for all models

def fit(x, X, Y, model_name):
    Torque = calc_torque(model_name, x, X)

    diff = np.sum((Torque - Y) ** 2)

    return diff


# In[7]:


def kl_divergence(p, q):
    return sum(p[i] * math.log(p[i]/q[i],2) for i in range(len(p)))


# In[8]:


df = pd.read_csv(r'C:\Users\Syeda Nimrah\Downloads\NimrahFreelancing\OMAE_2022\OMAE_2022\cleaned_data3_training.csv')
print(df)


# In[10]:


torque1 = Warren_Model(10,20,df['ROP'],df['RPM'],0.5,df['WOB'])
print(torque1)


# In[11]:


torque2 = Pessier_Model(0.5, df['WOB'], 30, 0.25)
print(torque2)


# In[12]:


torque3 = Patil_Model(12, df['RPM'], 0.75, 0.87, 0.5, df['WOB'])


# In[20]:


torque4 = Oklahoma_Model(df['WOB'], 0.5, df['ROP'], df['RPM'], 10, 10, 10, 10)
print(torque4)


# In[24]:


torque5 = Kuwait_Model(df['WOB'], 0.5, df['ROP'], 10, 20, df['RPM'])


# In[16]:


kl_pq = kl_divergence(torque1, torque2)
print('KL(torque1 || torque2): %.3f bits' % kl_pq)
kl_qp = kl_divergence(torque2, torque1)
print('KL(torque2 || torque1): %.3f bits' % kl_qp)


# In[17]:


kl_pq = kl_divergence(torque1, torque3)
print('KL(torque1 || torque3): %.3f bits' % kl_pq)
kl_qp = kl_divergence(torque3, torque1)
print('KL(torque3 || torque1): %.3f bits' % kl_qp)


# In[ ]:


kl_pq = kl_divergence(torque1, torque4)
print('KL(torque1 || torque4): %.3f bits' % kl_pq)
kl_qp = kl_divergence(torque4, torque1)
print('KL(torque4 || torque1): %.3f bits' % kl_qp)


# In[27]:


kl_pq = kl_divergence(torque1, torque5)
print('KL(torque1 || torque5): %.3f bits' % kl_pq)
kl_qp = kl_divergence(torque5, torque1)
print('KL(torque5 || torque1): %.3f bits' % kl_qp)


# In[26]:


kl_pq = kl_divergence(torque2, torque3)
print('KL(torque2 || torque3): %.3f bits' % kl_pq)
kl_qp = kl_divergence(torque3, torque2)
print('KL(torque3 || torque2): %.3f bits' % kl_qp)


# In[ ]:


kl_pq = kl_divergence(torque2, torque4)
print('KL(torque2 || torque4): %.3f bits' % kl_pq)
kl_qp = kl_divergence(torque4, torque2)
print('KL(torque4 || torque2): %.3f bits' % kl_qp)


# In[29]:


kl_pq = kl_divergence(torque2, torque5)
print('KL(torque2 || torque5): %.3f bits' % kl_pq)
kl_qp = kl_divergence(torque5, torque2)
print('KL(torque5 || torque2): %.3f bits' % kl_qp)


# In[ ]:


kl_pq = kl_divergence(torque3, torque4)
print('KL(torque3 || torque4): %.3f bits' % kl_pq)
kl_qp = kl_divergence(torque4, torque3)
print('KL(torque4 || torque3): %.3f bits' % kl_qp)


# In[30]:


kl_pq = kl_divergence(torque3, torque5)
print('KL(torque3 || torque5): %.3f bits' % kl_pq)
kl_qp = kl_divergence(torque5, torque3)
print('KL(torque5 || torque3): %.3f bits' % kl_qp)


# In[31]:


def cross_entropy_cost(yHat, y):
    if y == 1:
        return -np.log(yHat)
    else:
        return -np.log(1 - yHat)


# In[32]:


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# In[33]:


z = df['WOB']
h_z = sigmoid(z)

cost_1 = cross_entropy_cost(h_z, 1)
cost_0 = cross_entropy_cost(h_z, 0)


# In[34]:


print(cost_1)
print(cost_0)


# In[35]:


fig, ax = plt.subplots(figsize=(8,6))
plt.plot(h_z, cost_1, label='J(w) if y=1')
plt.plot(h_z, cost_0, label='J(w) if y=0')
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[40]:


def Itakuro_Saito_distance(p,q):
    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)
    D = kl_pq + kl_qp
    print(D)


# In[41]:


Itakuro_Saito_distance(torque1,torque2)


# In[42]:


Itakuro_Saito_distance(torque1,torque3)


# In[43]:


Itakuro_Saito_distance(torque1,torque5)


# In[44]:


Itakuro_Saito_distance(torque2,torque3)


# In[45]:


Itakuro_Saito_distance(torque2,torque5)


# In[46]:


Itakuro_Saito_distance(torque3,torque5)


# In[ ]:





# In[48]:


Itakuro_Saito_distance(torque2,torque3)


# In[ ]:





# In[6]:


import concurrent.futures
import numpy as np
import scipy.optimize
import time
import pickle
import sys


def optimize_models(x0, X, Y):

    models = [
        [x0, X, Y, "warren model"],
        [x0, X, Y, "pessier model"],
        [x0, X, Y, "patil model"],
        [x0, X, Y, "oklahoma model"],
        [x0, X, Y, "kuwait model"],
    ]

    my_models_list = ["warren model", "pessier model", "patil model", "oklahoma model", "kuwait model"]

    My_Optimizations_List = [
        "Nelder Mead", "Powell", "CG", "BFGS", "L_BFGS_B", "TNC", "COBYLA", "SLSQP",
        "trust_constr"
    ]
    costs = []  ## define a matrix

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for cost in executor.map(optimizations, models):
            ## Do all models in parallel for each optimization to speed up the System
            costs.append(cost)

    training_time = np.ones([len(models), len(My_Optimizations_List)]) * sys.float_info.max
    obj_fun = np.ones([len(models), len(My_Optimizations_List)]) * sys.float_info.max
    params = [[None for _ in range(0, len(My_Optimizations_List))] for _ in range(0, len(models))]

    # loop over models
    for i in range(0, len(models)):
        # loop over optimizations
        for j in range(0, len(My_Optimizations_List)):
            if costs[i][j][0].success == True:
                training_time[i][j] = costs[i][j][1]
                obj_fun[i][j] = costs[i][j][0].fun
                params[i][j] = costs[i][j][0].x
            else:
                print('training time: ' + str(costs[i][j][1]))
                print('obj fun: ' + str(costs[i][j][0].fun))
                print('params: ' + str(costs[i][j][0].x))
                print('message: ' + str(costs[i][j][0].message))

    ### sample code for loading
    #    file = open('test.pkl', 'rb')
    #    obj_1 = pickle.load(file)
    #    obj_2 = pickle.load(file)
    #    obj_3 = pickle.load(file)
    #    print(obj_1)
    #    print(obj_2)
    #    print(obj_3)
    #    file.close()

    ### end sample code for loading
    smallest = np.min(obj_fun)
    print(f"Best Optimization Method with the best model has the result: {smallest}")
    best_opt_index = np.argmin(obj_fun) % len(My_Optimizations_List)
    print("Best Optimization is: {best_optimization}".format(best_optimization=My_Optimizations_List[best_opt_index]))

    best_model_index = int(np.ceil((np.argmin(obj_fun) + 1) / len(My_Optimizations_List)) - 1)
#    print('obj_fun: ', obj_fun, '\n')
#    print('my_models_list: ', my_models_list, '\n')
#    print('best_model_index: ', best_model_index, '\n')
    print("Best Model is: {best_model}".format(best_model=my_models_list[best_model_index]))
    file = open('result' + time.strftime("%Y.%m.%d-%H_%M_%S") + '.pkl', 'wb')

    pickle.dump(my_models_list[best_model_index], file)
    pickle.dump(my_models_list, file)
    pickle.dump(training_time, file)
    pickle.dump(obj_fun, file)
    pickle.dump(params, file)

    file.close()


def optimizations(model):  ## All optimization methods are to be listed here

    ## Apply the following optimizing methods one after another

    x0 = model[0]
    X = model[1]
    Y = model[2]
    model_name = model[3]

    # print("running least squares method for model ", models)
    # t = time.process_time()
    # Params_least_squares = scipy.optimize.least_squares(models, x0, args=(X, Y, model_name))  # 1
    # t_least_squares = time.process_time() - t
    # print("least square finished in ", t_least_squares, " for model ", models)

    func_tol=2

    print("running Nelder Mead optimization method for model ", model_name)
    t = time.process_time()
    Params_Nelder_Mead = scipy.optimize.minimize(fit, x0, args=(X, Y, model_name), method='Nelder-Mead', tol=func_tol,
                                                 options={'return_all': False, 'maxiter':50000})  # 2
    t_Nelder_Mead = time.process_time() - t
    print("Nelder_Mead finished in ", t_Nelder_Mead, " for model ", model_name)

    print("running Powell optimization method", model_name)
    t = time.process_time()
    Params_Powell = scipy.optimize.minimize(fit, x0, args=(X, Y, model_name), method='Powell', tol=func_tol,
                                            options={'return_all': False, 'maxiter':50000})  # 3
    t_Powell = time.process_time() - t
    print("Powell finished in ", t_Powell, " for model ", model_name)

    print("running CG optimization method", model_name)
    t = time.process_time()
    Params_CG = scipy.optimize.minimize(fit, x0, args=(X, Y, model_name), method='CG', tol=func_tol,
                                        options={'return_all': False, 'maxiter':50000})  # 4
    t_CG = time.process_time() - t
    print("CG finished in ", t_CG, " for model ", model_name)

    print("running BFGS optimization method", model_name)
    t = time.process_time()
    Params_BFGS = scipy.optimize.minimize(fit, x0, args=(X, Y, model_name), method='BFGS', tol=func_tol,
                                          options={'return_all': False, 'maxiter':50000})  # 5
    t_BFGS = time.process_time() - t
    print("BFGS finished in ", t_BFGS, " for model ", model_name)

    print("running L_BFGS_B optimization method", model_name)
    t = time.process_time()
    Params_L_BFGS_B = scipy.optimize.minimize(fit, x0, args=(X, Y, model_name), method='L-BFGS-B', tol=func_tol, options={'maxiter':50000})  # 7
    t_L_BFGS_B = time.process_time() - t
    print("L_BFGS_B finished in ", t_L_BFGS_B, " for model ", model_name)

    print("running TNC optimization method", model_name)
    t = time.process_time()
    Params_TNC = scipy.optimize.minimize(fit, x0, args=(X, Y, model_name), method='TNC', tol=func_tol, options={'maxiter':50000})  # 8
    t_TNC = time.process_time() - t
    print("TNC finished in ", t_TNC, " for model ", model_name)

    print("running COBYLA optimization method", model_name)
    t = time.process_time()
    Params_COBYLA = scipy.optimize.minimize(fit, x0, args=(X, Y, model_name), method='COBYLA', tol=func_tol, options={'maxiter':50000})  # 9
    t_COBYLA = time.process_time() - t
    print("COBYLA finished in ", t_COBYLA, " for model ", model_name)

    print("running SLSQP optimization method", model_name)
    t = time.process_time()
    Params_SLSQP = scipy.optimize.minimize(fit, x0, args=(X, Y, model_name), method='SLSQP', tol=func_tol, options={'maxiter':50000})  # 10
    t_SLSQP = time.process_time() - t
    print("SLSQP finished in ", t_SLSQP, " for model ", model_name)

    print("running trust-constr optimization method", model_name)
    t = time.process_time()
    Params_trust_constr = scipy.optimize.minimize(fit, x0, args=(X, Y, model_name), method='trust-constr', tol=func_tol, options={'maxiter':50000})  # 11
    t_trust_constr = time.process_time() - t
    print("trust-constr finished in ", t_trust_constr, " for model ", model_name)

    # return np.array([[Params_least_squares, t_least_squares], [Params_Nelder_Mead, t_Nelder_Mead], [Params_Powell, t_Powell],
    return np.array([[Params_Nelder_Mead, t_Nelder_Mead], [Params_Powell, t_Powell],
                     [Params_CG, t_CG], [Params_BFGS, t_BFGS], [Params_L_BFGS_B, t_L_BFGS_B], [Params_TNC, t_TNC],
                     [Params_COBYLA, t_COBYLA], [Params_SLSQP, t_SLSQP], [Params_trust_constr, t_trust_constr]])


# import data in python

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import pickle
from itertools import groupby, islice
from multiprocessing import freeze_support, Process
import multiprocessing
import sys


def test_optimize():
    data = pd.read_csv("cleaned_data3_training.csv", names=['RPM','WOB','ROP','DEPTH','Claystone','Sandstone','Limestone','Torque'], skiprows=1)
    data['WOB'] = data['WOB'] * 2.2  # convert unit
    data['Torque'] = data['Torque'] * 737.5621  # convert unit
    print(data)

    # convert the data to same data used in model

    d = np.ones(len(data["ROP"])) * 12.25

    X = [data["ROP"].to_numpy(), data["RPM"].to_numpy() + 0.01, d, data["WOB"].to_numpy() + 0.01]

    ######
    # Startpunkte definieren
    # warren model
    C1 = 5
    C2 = 5
    # pessier model
    n = 5
    C3 = 5
    # patil model
    cb1 = 5
    ksb = 5
    kcb = 5
    # oklahoma model
    k = 5
    w = 5
    s = 5
    e = 5
    # kuwait model
    k2 = 5
    k3 = 5

    # collection of all initial conditions
    x0 = [C1, C2, n, C3, cb1, ksb, kcb, k, w, s, e, k2, k3]

    min_cost = []
    min_model = []
    min_params = []
    # Schichten = np.unique(dataDF)

    ind = -1
    Y = data["Torque"].to_numpy()

    optimize_models(x0, X, Y)


if __name__ == '__main__':
    test_optimize()


# In[ ]:




