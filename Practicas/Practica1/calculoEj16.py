import numpy as np  # Asignacion de variables

e = 0.0001
n = 0.4  # 0.1


# Definicion de la funcion
def f(array):
    return array[0] ** 2 + array[1] ** 2 + array[1] / 2 - 2


# Definicion del vector gradiente
def gradiente(array):
    return np.array([2 * array[0], 2 * array[1] + 0.5])


# Definicion de la condicion de corte
def corte(x_0, x_1):
    return abs(f(x_1) - f(x_0)) < e


# Multiplicacion del vector gradiente por el tamaño de paso
def mult_paso(x_1):
    return x_1 * n


# Calculo del minimo con el metodo de descenso del gradiente
def main():
    x_0 = np.array([10, 2])
    i = 1
    while True:
        print(f"Iteracion {i}")
        x_1 = x_0 - mult_paso(gradiente(x_0))
        if corte(x_0, x_1):
            return x_1
        x_0 = x_1
        i += 1


print(main())
