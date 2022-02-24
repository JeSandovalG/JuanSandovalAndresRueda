#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Solución del problema semanal
#Andrés Rueda López y Juan Esteban Sandoval Granados

#Importación de paquetes
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation as animation


# In[7]:


#Definición de variables
G = 6.67e-11
m_t = 5.9736E24
r_t = 6.3781E6
m_l = 0.07349E24
r_l =  1.7374e6
d =  3.844E8
w =  2.6617E-6

#Siguientes variables
Delta = G*m_t/d**3
Mu = m_l/m_t


#Esta es la función que ayudará a sacar la variable r_p que vamos a usar en la siguiente función.
def r_p(r_tilde, phi, t):
    return np.sqrt(1 + r_tilde**2 - 2*r_tilde*np.cos(phi - w*t))


#Esta es la función que devuelve las derivadas ya definidas previamente.
def derivadas(t,x):
    
    r = x[0]
    phi = x[1]
    Pr = x[2]
    Pphi = x[3]
    
    rp = r_p(r,phi,t)
    
    e1 = Pr
    e2 = Pphi/r**2
    e3 = Pphi**2/r**3 - Delta*(1/r**2 + Mu/rp**3*(r- np.cos(phi-w*t)))
    e4 = -(Delta*Mu*r/rp**3)*np.sin(phi-w*t)
    
    return np.array([e1,e2,e3,e4]) 




# In[8]:


#Definamos las condiciones inciales


####Esta velocidad inicial me esta generando problemas porque o se despega y se pasa de la luna o no despega de la tierra.
v_0 = 0.745/d
r_0 = r_t/d
theta = np.deg2rad(30)
phi = np.deg2rad(40)

Pr_0 = v_0*np.cos(theta - phi)
Pphi_0 = v_0*r_0*np.sin(theta - phi)


cond_ini = np.array([r_0, phi, Pr_0, Pphi_0])
#3 entrada es momento lineal 
#4 momento angular


# In[9]:


tmax = 259200
tspan = (0, tmax)

te = np.linspace(0,tmax, 1000)
sol = solve_ivp(derivadas, tspan, cond_ini, t_eval = te)





sol.y[1]


# In[11]:


plt.figure(figsize = (10,10),dpi=200)
plt.title("Trayectoria de un rocket a la luna")


t = np.linspace(0,tmax,10)

X_L = np.cos(w*t)
Y_L = np.sin(w*t)

plt.scatter(X_L,Y_L, s=100, color = "grey")

#Tierra
x = r_t/d
a = np.linspace(0, 2*np.pi)
plt.plot(x*np.cos(a), x*np.sin(a), "b")

X = sol.y[0]*np.cos(sol.y[1])
Y = sol.y[0]*np.sin(sol.y[1])
plt.plot(X,Y, "r")

plt.xlim(-1,1)
plt.ylim(-1,1)


# In[ ]:





# In[ ]:





# In[ ]:




