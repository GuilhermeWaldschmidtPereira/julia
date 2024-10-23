import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

def polar():
    v_x, v_y, v_x2, v_y2, v_x3, v_y3, v_x4, v_y4 = [], [], [], [], [], [], [], []
    a = 1.8
    b = 0.165
    theta = 0
    for i in range(150):
        x = a * np.cos(theta) * np.exp(b * theta)  # verde
        y = a * np.sin(theta) * np.exp(b * theta)
        v_x.append(x)
        v_y.append(y)
        
        x2 = a * np.cos(theta) * np.exp(b * theta)  # preto
        y2 = a * np.sin(theta) * np.exp(b * theta)
        v_x2.append(x2)
        v_y2.append(y2)

        x3 = a * np.cos(theta + np.pi) * np.exp(b * theta)  # vermelho
        y3 = a * np.sin(theta) * np.exp(b * theta)
        v_x3.append(x3)
        v_y3.append(y3)

        x4 = a * np.cos(theta + np.pi) * np.exp(b * theta)  # azul
        y4 = a * np.sin(theta) * np.exp(b * theta)
        v_x4.append(x4)
        v_y4.append(y4)

        theta += 0.1

    return v_x, v_y, v_x2, v_y2, v_x3, v_y3, v_x4, v_y4

vx, vy, vx2, vy2, vx3, vy3, vx4, vy4 = polar()
v2x = np.array(vx) + 20
v2y = np.array(vy) - 20
v2x2 = np.array(vx2) + 20
v2y2 = np.array(vy2) + 12
v2x3 = np.array(vx3) - 20
v2y3 = np.array(vy3) + 12
v2x4 = np.array(vx4) - 20
v2y4 = np.array(vy4) - 20

# Regressão para o primeiro conjunto
x, y = v2x[148:150], v2y[148:150]
data = pd.DataFrame({'x': x, 'y': y})
data['x2'] = data['x'] ** 2
model = sm.OLS(data['y'], sm.add_constant(data[['x', 'x2']])).fit()
print(model.params)

x_pred = np.arange(0, v2x[149], 0.01)
y_pred = model.predict(sm.add_constant(pd.DataFrame({'x': x_pred, 'x2': x_pred**2})))

# Regressão para o segundo conjunto
x, y = v2x2[148:150], v2y2[148:150]
data = pd.DataFrame({'x': x, 'y': y})
data['x2'] = data['x'] ** 2
model = sm.OLS(data['y'], sm.add_constant(data[['x', 'x2']])).fit()
print(model.params)

x_pred = np.arange(0, v2x2[149], 0.01)
y_pred = model.predict(sm.add_constant(pd.DataFrame({'x': x_pred, 'x2': x_pred**2})))

# Regressão para o terceiro conjunto
x, y = v2x3[148:150], v2y3[148:150]
data = pd.DataFrame({'x': x, 'y': y})
data['x2'] = data['x'] ** 2
model = sm.OLS(data['y'], sm.add_constant(data[['x', 'x2']])).fit()
print(model.params)

x_pred = np.arange(v2x3[149], 0, -0.01)
y_pred = model.predict(sm.add_constant(pd.DataFrame({'x': x_pred, 'x2': x_pred**2})))

# Regressão para o quarto conjunto
x, y = v2x4[148:150], v2y4[148:150]
data = pd.DataFrame({'x': x, 'y': y})
data['x2'] = data['x'] ** 2
model = sm.OLS(data['y'], sm.add_constant(data[['x', 'x2']])).fit()
print(model.params)

x_pred = np.arange(v2x4[149], 0, -0.01)
y_pred = model.predict(sm.add_constant(pd.DataFrame({'x': x_pred, 'x2': x_pred**2})))

# Salvando os dados em arquivos de texto
np.savetxt("txt_espiral_unitaria/fibv02_reg1.txt", np.column_stack((v2x, v2y)))
np.savetxt("txt_espiral_unitaria/fibv02_reg2.txt", np.column_stack((v2x2, v2y2)))
np.savetxt("txt_espiral_unitaria/fibv02_reg3.txt", np.column_stack((v2x3, v2y3)))
np.savetxt("txt_espiral_unitaria/fibv02_reg4.txt", np.column_stack((v2x4, v2y4)))

plt.plot(v2x, v2y, color='green')
plt.plot(v2x2, v2y2, color='black')
plt.plot(v2x3, v2y3, color='red')
plt.plot(v2x4, v2y4, color='blue')
plt.show()
