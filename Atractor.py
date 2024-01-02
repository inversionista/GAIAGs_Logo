#%%
# ************************************
#    Código del Atractar de Lorez
# ************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parámetros del sistema de Lorenz y condiciones iniciales
sigma, beta, rho = 10, 8/3, 28    # Parámetros estándar del atractor de Lorenz
u0, v0, w0 = 2, 1, 1             # Condiciones iniciales


# Punto de tiempo máximo y número total de puntos de tiempo
tmax, n = 100, 10000

# puntos en el tiempo
t = np.linspace(0, tmax, n)

# Ecuaciones diferenciales del sistema de Lorenz
def lorenz(X, t, sigma, beta, rho):
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

# Integra las ecuaciones de Lorenz en la malla temporal t
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))

# Grafica el atractor de Lorenz
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(f[:,0], f[:,1], f[:,2], lw=0.5, color= '#C427C3')
ax.set_xlabel("eje X")
ax.set_ylabel("eje Y")
ax.set_zlabel("eje Z")
ax.set_title("Atractor de Lorenz 3D")

plt.show()









#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parámetros del sistema de Lorenz y condiciones iniciales
sigma, beta, rho = 10, 8/3, 28
u0, v0, w0 = 0.5, 0.5, 10

# Punto de tiempo máximo y número total de puntos de tiempo
tmax, n = 100, 10000

# puntos en el tiempo
t = np.linspace(0, tmax, n)

# Ecuaciones diferenciales del sistema de Lorenz
def lorenz(X, t, sigma, beta, rho):
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

# Integra las ecuaciones de Lorenz en la malla temporal t
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))

# Grafica el atractor de Lorenz en 2D
fig, ax = plt.subplots()
ax.plot(f[:,0], f[:,1], lw=0.5, color= '#C427C3') # Using 'pink' as the color
ax.set_xlabel("eje X")
ax.set_ylabel("eje Y")
ax.set_title("Atractor de Lorenz 2D")

plt.show()



# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm

# Parámetros del sistema de Lorenz y condiciones iniciales
sigma, beta, rho = 10, 2.667, 28  # Parámetros estándar del atractor de Lorenz
u0, v0, w0 = 0, 1, 1.05  # Condiciones iniciales

# Punto de tiempo máximo y número total de puntos de tiempo
tmax, n = 100, 10000  # tmax es el tiempo final, n es el número de puntos

# Puntos en el tiempo
t = np.linspace(0, tmax, n)

# Ecuaciones diferenciales del sistema de Lorenz
def lorenz(X, t, sigma, beta, rho):
    u, v, w = X
    up = -sigma * (u - v)  # Ecuación diferencial para u
    vp = rho * u - v - u * w  # Ecuación diferencial para v
    wp = -beta * w + u * v  # Ecuación diferencial para w
    return up, vp, wp

# Integra las ecuaciones de Lorenz en la malla temporal t
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))

# Grafica el atractor de Lorenz en 2D
fig, ax = plt.subplots()
ax.plot(f[:,0], f[:,1], lw=0.5, color= '#C427C3')  # Usando el color de LOGO


# Calcula la media y la desviación estándar de los datos de 'u' para ajustar la curva normal
media_u = np.mean(f[:,0])
desviacion_u = np.std(f[:,0])


# Agregar una curva de distribución normal ajustada
x_normal = np.linspace(media_u - 3*desviacion_u, media_u + 3*desviacion_u, 100)
y_normal = norm.pdf(x_normal, media_u, desviacion_u)  # PDF de una distribución normal ajustada
y_normal_rescaled = y_normal * (max(f[:,1]) - min(f[:,1])) / max(y_normal)  # Escala la curva normal al rango de 'v'

# Resta un valor fijo a la curva normal para desplazarla hacia abajo
desplazamiento = 20  # Este valor lo puedes ajustar según necesites
y_normal_rescaled = y_normal_rescaled - desplazamiento  # Traslada la curva hacia abajo

ax.plot(x_normal, y_normal_rescaled, color='black', lw=2)  # Curva en el gráfico


# Ocultar los ejes y etiquetas de los ejes
ax.axis('off')

# Agregar texto en el centro de la gráfica
ax.text(0.52, 0.09, 'GAIAGs', transform=ax.transAxes, fontsize=42, va='center', ha='center', color='black')

plt.show()





# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del sistema de Lorenz y condiciones iniciales
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = 0, 1, 1.05
plano_normal = 32
desviación_curva_normal = 0.22*plano_normal

# Punto de tiempo máximo y número total de puntos de tiempo
tmax, n = 100, 10000

# Puntos en el tiempo
t = np.linspace(0, tmax, n)

# Ecuaciones diferenciales del sistema de Lorenz
def lorenz(X, t, sigma, beta, rho):
    u, v, w = X
    up = -sigma * (u - v)
    vp = rho * u - v - u * w
    wp = -beta * w + u * v
    return up, vp, wp

# Integra las ecuaciones de Lorenz en la malla temporal t
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))

# Grafica el atractor de Lorenz en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(f[:,0], f[:,1], f[:,2], lw=0.5, color='#C427C3')

# Crear una malla para la superficie de distribución normal
x = np.linspace(-plano_normal, plano_normal, 100)
y = np.linspace(-plano_normal, plano_normal, 100)
X, Y = np.meshgrid(x, y)
desviacion_estandar = desviación_curva_normal

Z = norm.pdf(X, scale=desviacion_estandar) * norm.pdf(Y, scale=desviacion_estandar)

# Ajustar la altura de la superficie de la distribución normal
factor_escalado = max(f[:,2]) / max(Z.flatten())
Z = Z * factor_escalado


desplazamiento_z = -50 #min(f[:,2]) - max(Z.flatten())  # Desplazamiento para colocar debajo del atractor
Z = Z + desplazamiento_z

# Graficar la superficie de la distribución normal con un mapa de colores
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Ocultar los ejes y etiquetas de los ejes
ax.set_axis_off()

# Agregar texto en el gráfico
ax.text2D(0.38, 0.22, 'GAIAGs', transform=ax.transAxes, fontsize=24, color='darkgray')

plt.show()




# %%import numpy as np
import matplotlib
#matplotlib.use('Agg')  # Usar backend para entorno sin GUI
matplotlib.use('TkAgg') 

import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Definir las ecuaciones diferenciales del sistema de Lorenz.
def lorenz_derivatives(state, t, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return dxdt, dydt, dzdt

# Parámetros iniciales y condiciones iniciales
sigma, beta, rho = 10.0, 2.667, 28.0
initial_state = [1.0, 1.0, 1.0]

# Crear una serie de tiempo
t = np.linspace(0, 40, 1000)

# Resolver el sistema de ecuaciones diferenciales
states = odeint(lorenz_derivatives, initial_state, t, args=(sigma, beta, rho))

# Configuración inicial de la figura y los ejes 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Función de animación que se llama secuencialmente
def animate(i):
    ax.clear()
    ax.plot(states[:i, 0], states[:i, 1], states[:i, 2], color='magenta', lw=1)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

# Crear animación usando FuncAnimation
ani = FuncAnimation(fig, animate, frames=len(t), interval=30)

# Guardar la animación
#ani.save('lorenz_attractor.mp4', writer='ffmpeg')

# Si quieres visualizar la figura estática como prueba
plt.show()  # Esto no funcionará en un entorno headless



# %%
# ***********************************************************
#    Ahora integración para Música con Aractor de de Lorez
# ***********************************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

# Parámetros del sistema de Lorenz y condiciones iniciales
sigma, beta, rho = 10, 8/3, 28    # Parámetros estándar del atractor de Lorenz
u0, v0, w0 = 2, 1, 1             # Condiciones iniciales


# Punto de tiempo máximo y número total de puntos de tiempo
tmax, n = 100, 10000

# puntos en el tiempo
t = np.linspace(0, tmax, n)

# Ecuaciones diferenciales del sistema de Lorenz
def lorenz(X, t, sigma, beta, rho):
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

# Integra las ecuaciones de Lorenz en la malla temporal t
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))

# Mapeo de los datos a frecuencias (puedes ajustar esto según sea necesario)
frequencies = (f[:,0] - min(f[:,0])) / (max(f[:,0]) - min(f[:,0])) * 440 + 440

# Grafica el atractor de Lorenz
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(f[:,0], f[:,1], f[:,2], lw=0.5, color= '#C427C3')
ax.set_xlabel("eje X")
ax.set_ylabel("eje Y")
ax.set_zlabel("eje Z")
ax.set_title("Atractor de Lorenz 3D")

plt.show()

# Crear un segmento de audio
audio = AudioSegment.silent(duration=0)

for freq in frequencies:
    # Genera un tono para cada punto de datos
    tone = Sine(freq).to_audio_segment(duration=50)
    audio += tone

# Exportar a un archivo
audio.export("lorenz_attractor_sound.wav", format="wav")
# %%



# %%
# ***********************************************************
#    Ahora integración para Música con Aractor de de Lorez
# ***********************************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

# Parámetros del sistema de Lorenz y condiciones iniciales
sigma, beta, rho = 10, 8/3, 28    # Parámetros estándar del atractor de Lorenz
u0, v0, w0 = 2, 1, 1             # Condiciones iniciales


# Punto de tiempo máximo y número total de puntos de tiempo
tmax, n = 100, 10000

# puntos en el tiempo
t = np.linspace(0, tmax, n)

# Ecuaciones diferenciales del sistema de Lorenz
def lorenz(X, t, sigma, beta, rho):
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

# Integra las ecuaciones de Lorenz en la malla temporal t
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))

# # Ejemplo: Escala de Do mayor
# scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # Frecuencias de las notas: Do, Re, Mi, Fa, Sol, La, Si

# # Escala Pentatónica Mayor de Do
# scale = [261.63, 293.66, 329.63, 392.00, 440.00]  # Frecuencias de las notas: Do, Re, Mi, Sol, La

# Escala Menor Armónica de La
scale = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 415.30]  # Frecuencias de las notas: La, Si, Do, Re, Mi, Fa, Sol♯

# Normaliza los datos
normalized_data = (f[:,0] - min(f[:,0])) / (max(f[:,0]) - min(f[:,0]))

# Mapea los datos normalizados a la escala musical
mapped_frequencies = [scale[int(note * (len(scale) - 1))] for note in normalized_data]

# Crear un segmento de audio
audio = AudioSegment.silent(duration=0)

# Duración de cada nota en milisegundos
note_duration = 68

for freq in mapped_frequencies:
    tone = Sine(freq).to_audio_segment(duration=note_duration)
    audio += tone

# Exportar a un archivo
audio.export("lorenz_attractor_melody_7x  .wav", format="wav")
# %%
