import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

def polar():
    v_x, v_y, v_x2, v_y2, v_x3, v_y3, v_x4, v_y4 = [], [], [], [], [], [], [], []
    r = 0.0
    theta = 0
    theta_1 = 0
    theta2 = 0
    theta3 = 0
    theta4 = 0
    n = 2
    for i in range(200):
        x = (r * theta) * np.cos(theta_1 + np.pi / 4) / n  # verde
        y = (r * theta) * np.sin(theta_1 + np.pi / 4) / n  # verde
        x2 = (r * theta) * np.cos(theta_1 + np.pi / 1.4) / n  # preto
        y2 = (r * theta) * np.sin(theta_1 + np.pi / 1.4) / n  # preto
        x3 = (r * theta) * np.cos(theta_1 - np.pi / 1.3) / n  # vermelho
        y3 = (r * theta) * np.sin(theta_1 - np.pi / 1.3) / n  # vermelho
        x4 = (r * theta) * np.cos(theta_1 - np.pi / 3.75) / n  # azul
        y4 = (r * theta) * np.sin(theta_1 - np.pi / 3.75) / n  # azul
        
        v_x.append(x)
        v_y.append(y)
        v_x2.append(x2)
        v_y2.append(y2)
        v_x3.append(x3)
        v_y3.append(y3)
        v_x4.append(x4)
        v_y4.append(y4)

        theta += 0.0122
        theta_1 += 0.1
        theta2 += 0.1
        theta3 += 0.1
        theta4 += 0.1
        r += 0.067

    return v_x, v_y, v_x2, v_y2, v_x3, v_y3, v_x4, v_y4

vx, vy, vx2, vy2, vx3, vy3, vx4, vy4 = polar()

v2x = np.array(vx) + 16.5
v2y = np.array(vy) - 16.5
v2x2 = np.array(vx2) + 16.5
v2y2 = np.array(vy2) + 16.5
v2x3 = np.array(vx3) - 16.5
v2y3 = np.array(vy3) + 16.5
v2x4 = np.array(vx4) - 16.5
v2y4 = np.array(vy4) - 16.5

# Plot das espirais
plt.plot(v2x, v2y, label='', color='green')
plt.plot(v2x2, v2y2, label='', color='black')
plt.plot(v2x3, v2y3, label='', color='red')
plt.plot(v2x4, v2y4, label='', color='blue')

# Regressão para o conjunto 1
x, y = v2x[190:200], v2y[190:200]
data = pd.DataFrame({'x': x, 'y': y})
data['x2'] = data['x'] ** 2
model = sm.OLS(data['y'], sm.add_constant(data[['x', 'x2']])).fit()
print(model.params)

x_pred = np.arange(0, 20, 0.1)
y_pred = model.predict(sm.add_constant(pd.DataFrame({'x': x_pred, 'x2': x_pred**2})))
plt.plot(x_pred, y_pred, color='black', label='')

# Regressão para o conjunto 2
x, y = v2x2[195:200], v2y2[195:200]
data = pd.DataFrame({'x': x, 'y': y})
data['x2'] = data['x'] ** 2
model = sm.OLS(data['y'], sm.add_constant(data[['x', 'x2']])).fit()
print(model.params)

x_pred = np.arange(2, 6, 0.1)
y_pred = model.predict(sm.add_constant(pd.DataFrame({'x': x_pred, 'x2': x_pred**2})))
plt.plot(x_pred, y_pred, color='yellow', label='')

# Regressão para o conjunto 3
x, y = v2x3[195:200], v2y3[195:200]
data = pd.DataFrame({'x': x, 'y': y})
data['x2'] = data['x'] ** 2
model = sm.OLS(data['y'], sm.add_constant(data[['x', 'x2']])).fit()
print(model.params)

x_pred = np.arange(-20, 0, 0.1)
y_pred = model.predict(sm.add_constant(pd.DataFrame({'x': x_pred, 'x2': x_pred**2})))
plt.plot(x_pred, y_pred, color='black', label='')

# Regressão para o conjunto 4
x, y = v2x4[195:200], v2y4[195:200]
data = pd.DataFrame({'x': x, 'y': y})
data['x2'] = data['x'] ** 2
model = sm.OLS(data['y'], sm.add_constant(data[['x', 'x2']])).fit()
print(model.params)

x_pred = np.arange(-1.5, 0.0, 0.1)
y_pred = model.predict(sm.add_constant(pd.DataFrame({'x': x_pred, 'x2': x_pred**2})))
plt.plot(x_pred, y_pred, color='black', label='')

# Exibir gráfico final
plt.show()

# Salvando os dados em arquivos de texto
np.savetxt("espiral2.txt", np.column_stack((v2x, v2y, v2x2, v2y2, v2x3, v2y3, v2x4, v2y4)))
