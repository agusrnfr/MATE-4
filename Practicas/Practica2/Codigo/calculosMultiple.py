import pandas as pd
import numpy as np

# Carga el archivo CSV
ruta_archivo = "punto14.csv"
df = pd.read_csv(ruta_archivo)


def calculos_auxiliares(df):
    # Valores de X1, X2, X3 y Y
    x1 = df["x1"].sum()
    x2 = df["x2"].sum()
    x3 = df["x3"].sum()
    y = df["y"].sum()

    # Longitud de la tabla
    n = len(df)

    # Sumatorias necesarias
    sumX1X2 = (df["x1"] * df["x2"]).sum()
    sumX1X3 = (df["x1"] * df["x3"]).sum()
    sumX2X3 = (df["x2"] * df["x3"]).sum()
    sumX1Y = (df["x1"] * df["y"]).sum()
    sumX2Y = (df["x2"] * df["y"]).sum()
    sumX3Y = (df["x3"] * df["y"]).sum()
    sumX1_2 = (df["x1"] ** 2).sum()
    sumX2_2 = (df["x2"] ** 2).sum()
    sumX3_2 = (df["x3"] ** 2).sum()

    # Construcción de las matrices para los cálculos
    X = np.column_stack((np.ones(n), df["x1"], df["x2"], df["x3"]))
    Y = df["y"]

    # Cálculo de los coeficientes B0, B1, B2, B3 usando la fórmula de regresión múltiple
    B = np.linalg.inv(X.T @ X) @ X.T @ Y

    B0 = B[0]  # Ordenada al origen
    B1 = B[1]  # Coeficiente de x1
    B2 = B[2]  # Coeficiente de x2
    B3 = B[3]  # Coeficiente de x3

    # Predicciones del modelo
    Y_pred = X @ B

    # Cálculo de Syy (variación total)
    Syy = ((df["y"] - y / n) ** 2).sum()

    # Cálculo de Ssr (variación del residuo)
    Ssr = ((df["y"] - Y_pred) ** 2).sum()

    # Cálculo de R2
    R2 = 1 - Ssr / Syy

    R2a = 1 - (1 - R2) * ((n - 3) / (n - 3 - 1))

    # Cálculo de la correlación r (para una regresión múltiple esto es más complejo)
    r = np.sqrt(R2a)

    # Cálculo de la varianza
    o2 = Ssr / (n - 4)

    return {
        "sumX1": x1,
        "sumX2": x2,
        "sumX3": x3,
        "sumY": y,
        "sumX1_2": sumX1_2,
        "sumX2_2": sumX2_2,
        "sumX3_2": sumX3_2,
        "sumX1X2": sumX1X2,
        "sumX1X3": sumX1X3,
        "sumX2X3": sumX2X3,
        "sumX1Y": sumX1Y,
        "sumX2Y": sumX2Y,
        "sumX3Y": sumX3Y,
        "y/": y / n,
        "x1/": x1 / n,
        "x2/": x2 / n,
        "x3/": x3 / n,
        "n": n,
        "Syy": Syy,
        "Ssr": Ssr,
        "B0": B0,
        "B1": B1,
        "B2": B2,
        "B3": B3,
        "R2": R2,
        "R2a": R2a,
        "r": r,
        "o2": o2,
    }


# Grafica la dispersión de los datos en 3D (X1, X2) y su hiperplano de regresión
def grafica_dispercion(df, B0, B1, B2, B3):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Valores de X1, X2, X3 y Y
    x1 = df["x1"]
    x2 = df["x2"]
    y = df["y"]

    # Crear figura en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Grafica de dispersión (solo para x1, x2 por simplicidad visual)
    ax.scatter(x1, x2, y, color="blue", label="Datos")

    # Crear malla de puntos para graficar el hiperplano de regresión
    x1_surf, x2_surf = np.meshgrid(
        np.linspace(x1.min(), x1.max(), 20), np.linspace(x2.min(), x2.max(), 20)
    )
    y_surf = B0 + B1 * x1_surf + B2 * x2_surf + B3 * np.mean(df["x3"])

    # Grafica el hiperplano de regresión
    ax.plot_surface(
        x1_surf,
        x2_surf,
        y_surf,
        color="red",
        alpha=0.5,
        label="Hiperplano de regresión",
    )

    # Configuración de la gráfica
    ax.set_xlabel("X1 (Variable 1)")
    ax.set_ylabel("X2 (Variable 2)")
    ax.set_zlabel("Y (Variable dependiente)")
    plt.title("Dispersión de los datos y hiperplano de regresión")
    plt.show()


# Ejecución
datos = calculos_auxiliares(df)
print(datos)
grafica_dispercion(df, datos["B0"], datos["B1"], datos["B2"], datos["B3"])
