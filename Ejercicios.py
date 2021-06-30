import numpy as np

print("Ejercicio 88 \nConsiderando 2 vectores A y B, escriba el equivalente einsum de inner, outside, sum, y función mul")
A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)
print("Matriz A:\n", A)
np.einsum('i,i->i', A, B)
print("Matriz A:\n", A,"\nMatriz B:\n",B)
np.einsum('i,i', A, B)
print("Matriz A:\n", A,"\nMatriz B:\n",B)
np.einsum('i,j', A, B)
print("Matriz A:\n", A,"\nMatriz B:\n",B)
print("Ejercicio 89 \nConsiderando una ruta descrita por dos vectores (X, Y), cómo muestrearla usando muestras equidistantes")
phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)
dr = (np.diff(x)**2 + np.diff(y)**2)**.5
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)
r_int = np.linspace(0, r.max(), 200)
x_int = np.interp(r_int, r, x)
y_int = np.interp(r_int, r, y)

print("Ejercicio 90 \nDado un número entero ny una matriz 2D X, seleccione de X las filas que pueden ser interpretado como extractos de una distribución multinomial con n grados, es decir, las filas que solo contienen números enteros y que suman n")
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
[2.0, 0.0, 1.0, 1.0],
[1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)

print(X[M])

