import pandas as pd

# Carga el archivo Excel
ruta_archivo = "punto13.csv"
df = pd.read_csv(ruta_archivo)


def calculos_auxiliares(df):
    # Valores de X e Y
    x = df["x"].sum()
    y = df["y"].sum()

    # Sumatoria de la multiplicacion entre dias y precio
    sumPor = (df["x"] * df["y"]).sum()

    # Longitud de la tabla
    n = len(df)

    # Calculo de Sxy
    Sxy = sumPor - (x * y) / n

    # Calculo de Sxx
    Sxx = (df["x"] ** 2).sum() - (x**2) / n

    # Calculo de Syy
    Syy = (df["y"] ** 2).sum() - (y**2) / n

    # Calculo de la pendiente (Estimador B1)
    B1 = Sxy / Sxx

    # Calculo de la ordenada al origen (Estimador B0)
    B0 = y / n - B1 * x / n

    # Calculo de Ssr
    Ssr = Syy - B1 * Sxy

    # Calculo de R2
    R2 = 1 - Ssr / Syy

    # Calculo de r
    r = Sxy / ((Sxx * Syy) ** 0.5)

    # Calculo de la varianza
    o2 = Ssr / (n - 2)

    return {
        "sumPor": sumPor,
        "sumY": y,
        "sumX": x,
        "y/": y / n,
        "x/": x / n,
        "n": n,
        "Syy": Syy,
        "Ssr": Ssr,
        "Sxx": Sxx,
        "Sxy": Sxy,
        "B0": B0,
        "B1": B1,
        "R2": R2,
        "r": r,
        "o2": o2,
    }


# Grafica la dispersión de los datos y su recta de regresión
def grafica_dispercion(df, B0, B1):
    import matplotlib.pyplot as plt

    # Valores de X e Y
    x = df["x"]
    y = df["y"]

    # Grafica de dispersión
    plt.scatter(x, y, color="blue", label="Datos")

    # Grafica de la recta de regresión
    plt.plot(x, B0 + B1 * x, color="red", label="Recta de regresión")

    # Configuración de la gráfica
    plt.xlabel("Tiempo en días")
    plt.ylabel("Frescura")
    plt.title("Dispersión de los datos y recta de regresión")
    plt.legend()
    plt.show()


datos = calculos_auxiliares(df)
print(datos)
grafica_dispercion(df, datos["B0"], datos["B1"])
